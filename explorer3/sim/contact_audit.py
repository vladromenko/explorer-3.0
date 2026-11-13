"""Read-only contact measurements, including transients between control frames."""
import numpy as np


class ContactAudit:
    def __init__(self, model):
        self.cube=model.body('task_cube').id
        self.minimum_distance_m=0.
        self.maximum_mimic_error_rad=0.
        self.substeps=0
        self.lifted=False
        self.worst_contact=None
        self.minimum_carry_height_m=float('inf')
        self.linked=model.nq==18

    def __call__(self, model, data):
        self.substeps+=1
        self.lifted=self.lifted or bool(data.xpos[self.cube,2]>.1)
        # These teachers finish lifting at 8 s and only open at 13 s.
        if 8. <= data.time <= 13.:
            self.minimum_carry_height_m=min(self.minimum_carry_height_m,float(data.xpos[self.cube,2]))
        for contact in data.contact:
            if self.cube in model.geom_bodyid[[contact.geom1,contact.geom2]]:
                if contact.dist<self.minimum_distance_m:
                    self.minimum_distance_m=float(contact.dist)
                    self.worst_contact={'time_s':float(data.time),
                        'geoms':[model.geom(g).name or int(g) for g in (contact.geom1,contact.geom2)],
                        'bodies':[model.body(model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]}
        if self.linked:
            error=float(np.max(np.abs(data.qpos[6:11]-data.qpos[5]*np.array([1,1,-1,-1,1]))))
            self.maximum_mimic_error_rad=max(self.maximum_mimic_error_rad,error)

    def report(self):
        return {'physics_substeps_observed':self.substeps,'lifted_above_100mm':self.lifted,
                'worst_contact':self.worst_contact,
                'minimum_carry_height_m':self.minimum_carry_height_m,
                'maximum_cube_penetration_mm':-self.minimum_distance_m*1000,
                'maximum_mimic_error_rad':self.maximum_mimic_error_rad if self.linked else None}
