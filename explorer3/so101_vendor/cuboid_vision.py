# Copied from so101_autonomous_pick_place/src/so101_pick_place/geometry/cuboid_vision.py (2026-11-16).
# Changes: the result is returned as a dict instead of the source project's CubeGeometry (whose
# module pulls in that project's vision types); added fit_upright_cuboid (same silhouette residual,
# separate width / length / height) for non-cube objects. The original fit is kept unchanged.
"""Fit an upright metric cube to complete camera silhouettes on a known table.

All eight vertices are projected. Oblique views may therefore include side
faces: their apparent width is not mistaken for a top-plane edge length.
"""
from numbers import Real

import cv2
import numpy as np
from scipy.optimize import least_squares

from .table_projection import TableProjection


def cube_vertices(xy,side,yaw,table_z=0.):
    cosine,sine=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[cosine,-sine],[sine,cosine]])
    square=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*side
    square=square@rotation.T+np.asarray(xy)
    return np.vstack((np.column_stack((square,np.full(4,table_z))),np.column_stack((square,np.full(4,table_z+side)))))


def project_vertices(vertices,calibration):
    pose=np.asarray(calibration["camera_from_robot"],float)
    camera=vertices@pose[:3,:3].T+pose[:3,3]
    if np.any(camera[:,2]<=.015):
        raise ValueError("cube projection is behind or too close to the camera")
    rotation,_=cv2.Rodrigues(pose[:3,:3])
    points,_=cv2.projectPoints(vertices,rotation,pose[:3,3],np.asarray(calibration["camera_matrix"],float),np.asarray(calibration["distortion"],float))
    return points.reshape(-1,2)


def outline_samples(polygon,count=64):
    polygon=np.asarray(polygon).reshape(-1,2)
    closed=np.vstack((polygon,polygon[0]))
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    points=np.linspace(0.,distance[-1],count,endpoint=False)
    return np.column_stack([np.interp(points,distance,closed[:,index]) for index in (0,1)])


def fit_upright_cube(observations,*,minimum_side_m=.015,maximum_side_m=.065):
    """observations: nonempty list of (complete object mask, camera calibration)."""
    if not observations:
        raise ValueError("at least one complete cube silhouette is required")
    table_heights=[calibration.get("table_z_m",0.) for _,calibration in observations]
    if any(not isinstance(height,Real) or isinstance(height,(bool,np.bool_)) or not np.isfinite(height) for height in table_heights):
        raise ValueError("each cube observation requires a finite numeric table plane")
    table_z=float(table_heights[0])
    # Every view describes the same base-frame plane; allow only numeric rounding.
    if not np.allclose(table_heights,table_z,rtol=0.,atol=1e-6):
        raise ValueError("cube observations use inconsistent table planes")
    prepared=[]
    for mask,calibration in observations:
        mask=np.asarray(mask)
        if mask.ndim!=2 or min(mask.shape)<3 or not np.all(np.isfinite(mask)):
            raise ValueError("cube silhouette must be a finite two-dimensional mask")
        mask=np.uint8(mask>0)
        if np.any(mask[:2]) or np.any(mask[-2:]) or np.any(mask[:,:2]) or np.any(mask[:,-2:]):
            raise ValueError("a clipped cube silhouette cannot supply metric geometry")
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        if count<=1:
            raise ValueError("cube silhouette is empty")
        largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        area=int(stats[largest,cv2.CC_STAT_AREA])
        # Ignore sparse segmentation dust, at most 1% of the object (or 8 pixels).
        # A second object or a substantial detached fragment is ambiguous geometry.
        if np.count_nonzero(mask)-area>max(8.,.01*area):
            raise ValueError("cube silhouette contains multiple significant components")
        mask=np.uint8(labels==largest)
        # Only enclosed holes of at most nine pixels and a 4x4 bounding box
        # count as raster noise. Keep larger holes and every boundary notch:
        # the metric silhouette residual, rather than a convexified mask,
        # must explain the actual observed foreground.
        background_count,background_labels,background_stats,_=cv2.connectedComponentsWithStats(1-mask,connectivity=8)
        outside_label=int(background_labels[0,0])
        for label in range(1,background_count):
            small=(background_stats[label,cv2.CC_STAT_AREA]<=9
                   and background_stats[label,cv2.CC_STAT_WIDTH]<=4
                   and background_stats[label,cv2.CC_STAT_HEIGHT]<=4)
            if label!=outside_label and small:
                mask[background_labels==label]=1
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        outer=max(contours,key=cv2.contourArea).astype(np.float32)
        if cv2.contourArea(outer)<150.:
            raise ValueError("cube silhouette has insufficient image resolution")
        inside=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        outside=cv2.distanceTransform(1-mask,cv2.DIST_L2,5)
        signed=np.where(mask>0,inside-.5,-outside+.5).astype(np.float32)
        # Include hole boundaries as well as the raw external silhouette. A
        # model cannot explain a large missing interior merely by matching its
        # outer edge. Each contour receives a full set of boundary samples.
        boundary=np.concatenate([outline_samples(contour) for contour in contours])
        prepared.append((outer,boundary,signed,calibration))
    outer,_,_,calibration=prepared[0]
    rectangle=cv2.minAreaRect(outer)
    projection=TableProjection(np.asarray(calibration["camera_matrix"]),np.asarray(calibration["distortion"]),np.asarray(calibration["camera_from_robot"]),table_z+.015)
    initial_xy=np.asarray(projection.pixel_to_table(rectangle[0]).xyz_m[:2])
    corners=cv2.boxPoints(rectangle)
    world_corners=np.asarray([projection.pixel_to_table(tuple(point)).xyz_m for point in corners])
    initial_side=float(np.clip(np.median(np.linalg.norm(np.roll(world_corners,-1,axis=0)-world_corners,axis=1)),minimum_side_m+.001,maximum_side_m-.001))
    edge=world_corners[1,:2]-world_corners[0,:2]
    initial_yaw=float(np.arctan2(edge[1],edge[0]))

    def residual(parameters):
        xy=parameters[:2];side=np.exp(parameters[2]);yaw=parameters[3]
        vertices=cube_vertices(xy,side,yaw,table_z)
        errors=[]
        for _,boundary,signed,calibration in prepared:
            predicted=cv2.convexHull(project_vertices(vertices,calibration).astype(np.float32))
            sample=outline_samples(predicted)
            x=np.clip(sample[:,0],0.,signed.shape[1]-1.001)
            y=np.clip(sample[:,1],0.,signed.shape[0]-1.001)
            a=x.astype(int);b=y.astype(int);u=x-a;v=y-b
            values=(1-u)*(1-v)*signed[b,a]+u*(1-v)*signed[b,a+1]+(1-u)*v*signed[b+1,a]+u*v*signed[b+1,a+1]
            errors.extend(values)
            errors.extend(cv2.pointPolygonTest(predicted,(float(p[0]),float(p[1])),True) for p in boundary)
        return np.asarray(errors)

    steps=np.array([.00005,.00005,.0005,.001])
    def jacobian(parameters):
        columns=[]
        for index,step in enumerate(steps):
            delta=np.zeros(4);delta[index]=step
            columns.append((residual(parameters+delta)-residual(parameters-delta))/(2*step))
        return np.column_stack(columns)

    lower=np.r_[initial_xy-.08,np.log(minimum_side_m),initial_yaw-np.pi]
    upper=np.r_[initial_xy+.08,np.log(maximum_side_m),initial_yaw+np.pi]
    solutions=[]
    for yaw in initial_yaw+np.arange(4)*np.pi/8.:
        initial=np.r_[initial_xy,np.log(initial_side),yaw]
        try:
            result=least_squares(residual,initial,jac=jacobian,bounds=(lower,upper),loss="soft_l1",f_scale=1.5,x_scale="jac",max_nfev=250)
            solutions.append(result)
        except (ValueError,cv2.error):
            pass
    if not solutions:
        raise ValueError("no valid calibrated cube silhouette projection was found")
    best=min(solutions,key=lambda item:np.mean(residual(item.x)**2))
    error=residual(best.x)
    rmse=float(np.sqrt(np.mean(error**2)))
    q95=float(np.quantile(np.abs(error),.95))
    side=float(np.exp(best.x[2]));yaw=float(best.x[3]);xy=best.x[:2]
    if rmse>2.5 or q95>5.:
        raise ValueError(f"cube silhouette is incomplete or inconsistent with an upright cube: silhouette error={rmse:.2f}px")
    if min(side-minimum_side_m,maximum_side_m-side)<.0003:
        raise ValueError("cube size reaches the physical model boundary")
    raw_jacobian=jacobian(best.x)
    norms=np.maximum(np.linalg.norm(raw_jacobian,axis=0),1e-12)
    singular=np.linalg.svd(raw_jacobian/norms,compute_uv=False)
    if singular[-1]<1e-5*singular[0]:
        raise ValueError("the silhouettes do not independently constrain the cube pose and size")
    covariance=np.linalg.pinv(raw_jacobian.T@raw_jacobian)*max(.5,rmse)**2
    sigma=np.sqrt(np.maximum(0.,np.diag(covariance)))
    if max(sigma[:2])>.002 or sigma[2]*side>.002:
        raise ValueError("cube metric pose or size is not sufficiently observable")
    vertices=cube_vertices(xy,side,yaw,table_z)
    axes=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    cube={"top_center_m":np.r_[xy,table_z+side].tolist(),"center_m":np.r_[xy,table_z+side/2.].tolist(),"side_m":side,"yaw_rad":yaw,"axes":axes.tolist()}
    return cube,{"method":"known table plane and all eight metric cube vertices fitted to camera silhouettes","view_count":len(observations),"silhouette_rmse_px":rmse,"silhouette_q95_px":q95,"observed_mask_convexified":False,"position_sigma_m":sigma[:2].tolist(),"side_sigma_m":float(sigma[2]*side),"size_m":side}


def cuboid_vertices(xy,sides,yaw,table_z=0.):
    w,l,h=sides
    cosine,sine=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[cosine,-sine],[sine,cosine]])
    square=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*np.array([w,l])
    square=square@rotation.T+np.asarray(xy)
    return np.vstack((np.column_stack((square,np.full(4,table_z))),np.column_stack((square,np.full(4,table_z+h)))))


def fit_upright_cuboid(mask,calibration,*,minimum_m=.008,maximum_m=.25,height_m=None):
    """One complete silhouette -> upright box (x, y, width, length, height, yaw), same residual as the
    cube fit. With a single top-down view the height is weakly observable: pass height_m when known;
    otherwise it is fitted and its sigma is reported (not hidden)."""
    mask=np.uint8(np.asarray(mask)>0)
    if np.any(mask[:2]) or np.any(mask[-2:]) or np.any(mask[:,:2]) or np.any(mask[:,-2:]):
        raise ValueError("a clipped silhouette cannot supply metric geometry")
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    if count<=1:
        raise ValueError("silhouette is empty")
    mask=np.uint8(labels==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    outer=max(contours,key=cv2.contourArea).astype(np.float32)
    if cv2.contourArea(outer)<150.:
        raise ValueError("silhouette has insufficient image resolution")
    inside=cv2.distanceTransform(mask,cv2.DIST_L2,5);outside=cv2.distanceTransform(1-mask,cv2.DIST_L2,5)
    signed=np.where(mask>0,inside-.5,-outside+.5).astype(np.float32)
    boundary=outline_samples(outer,96)
    table_z=float(calibration.get("table_z_m",0.))
    rectangle=cv2.minAreaRect(outer)
    projection=TableProjection(np.asarray(calibration["camera_matrix"]),np.asarray(calibration["distortion"]),np.asarray(calibration["camera_from_robot"]),table_z+.015)
    initial_xy=np.asarray(projection.pixel_to_table(rectangle[0]).xyz_m[:2])
    world=np.asarray([projection.pixel_to_table(tuple(p)).xyz_m for p in cv2.boxPoints(rectangle)])
    a=np.linalg.norm(world[1]-world[0]);b=np.linalg.norm(world[2]-world[1])
    edge=world[1,:2]-world[0,:2];initial_yaw=float(np.arctan2(edge[1],edge[0]))
    fixed_h=height_m

    def unpack(p):
        h=fixed_h if fixed_h is not None else np.exp(p[5])
        return p[:2],(np.exp(p[2]),np.exp(p[3]),h),p[4]

    def residual(p):
        xy,sides,yaw=unpack(p)
        predicted=cv2.convexHull(project_vertices(cuboid_vertices(xy,sides,yaw,table_z),calibration).astype(np.float32))
        sample=outline_samples(predicted)
        x=np.clip(sample[:,0],0.,signed.shape[1]-1.001);y=np.clip(sample[:,1],0.,signed.shape[0]-1.001)
        i=x.astype(int);j=y.astype(int);u=x-i;v=y-j
        values=(1-u)*(1-v)*signed[j,i]+u*(1-v)*signed[j,i+1]+(1-u)*v*signed[j+1,i]+u*v*signed[j+1,i+1]
        return np.r_[values,[cv2.pointPolygonTest(predicted,(float(q[0]),float(q[1])),True) for q in boundary]]

    initial=np.r_[initial_xy,np.log(np.clip([a,b],minimum_m*1.1,maximum_m*.9)),initial_yaw]
    lower=np.r_[initial_xy-.08,np.log(minimum_m),np.log(minimum_m),initial_yaw-np.pi]
    upper=np.r_[initial_xy+.08,np.log(maximum_m),np.log(maximum_m),initial_yaw+np.pi]
    if fixed_h is None:
        initial=np.r_[initial,np.log(np.clip(min(a,b),minimum_m*1.1,maximum_m*.9))]
        lower=np.r_[lower,np.log(minimum_m)];upper=np.r_[upper,np.log(maximum_m)]
    best=least_squares(residual,initial,bounds=(lower,upper),loss="soft_l1",f_scale=1.5,x_scale="jac",max_nfev=400,diff_step=1e-4)
    error=residual(best.x);rmse=float(np.sqrt(np.mean(error**2)))
    J=best.jac;cov=np.linalg.pinv(J.T@J)*max(.5,rmse)**2;sigma=np.sqrt(np.maximum(0.,np.diag(cov)))
    xy,(w,l,h),yaw=unpack(best.x)
    w,l=sorted([float(w),float(l)])
    return {"center_m":[float(xy[0]),float(xy[1]),table_z+float(h)/2],"width_m":w,"length_m":l,"height_m":float(h),
            "yaw_rad":float(yaw),"silhouette_rmse_px":rmse,"height_measured":fixed_h is None,
            "height_sigma_m":(float(sigma[5]*h) if fixed_h is None else None),
            "side_sigma_m":float(max(sigma[2]*w,sigma[3]*l))}
