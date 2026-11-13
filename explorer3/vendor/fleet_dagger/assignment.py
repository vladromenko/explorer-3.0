"""Exact method excerpt from iflb/parallel_experiment.py; see LICENSE/provenance."""
import numpy as np

class Assignment:
    def assign_humans(self, env_priorities):
        '''
        Fleet-DAgger assignment given a prioritization
        '''
        # Make sure there is at most 1 human each robot is assigned to
        assert 0 <= self.assignments.sum(1).max() <= 1
        
        # Humans that won't be reallocated this timestep. This includes humans that still have time before they can switch
        # and humans that are already assigned to high priority robots.
        humans_to_keep = set()

        # Find number of humans that could possibly be reassigned
        max_humans_to_assign = 0
        for i in range(self.exp_cfg.num_humans):
            # Reassign human if they are not yet assigned or
            if not np.sum(self.assignments[:, i]):
                max_humans_to_assign += 1
            # their time is up and they are not resetting a hard failure
            elif self.human_timers[i] >= self.exp_cfg.min_int_time and not self.blocked_envs[np.argmax(self.assignments[:,i])]:
                max_humans_to_assign += 1
            # otherwise keep them where they are
            else:
                humans_to_keep.add(i)

        assignment_count = 0 # next worst robot index
        human_idx = 0 # number of reassignable humans accounted for
        # Use env priorities to identify which humans should not be reassigned
        while human_idx < max_humans_to_assign and assignment_count < len(env_priorities):
            # check the next worst robot.
            env_assign_idx = env_priorities[assignment_count]
            # If someone is already assigned to this env...
            if self.assignments[env_assign_idx].sum():
                # find out who
                assigned_human = np.argmax(self.assignments[env_assign_idx])
                assignment_count += 1
                # This high priority robot already has a human assigned to it, so we will not reallocate assigned_human
                humans_to_keep.add(assigned_human)
                # If assigned_human is about to time out though, we will allocate a human to it, it'll just be the same
                # human since no need to swap in a new human in this high priority env that still needs help
                if self.human_timers[assigned_human] >= self.exp_cfg.min_int_time and not self.blocked_envs[np.argmax(self.assignments[:,i])]:
                    human_idx += 1
            else:
                # If no human assigned, we will be assigning a human, so move to next human.
                human_idx += 1

        # Actually assign humans
        assignment_count = 0
        human_idx = 0
        while human_idx < self.exp_cfg.num_humans:
            # If human is assigned but its allocation won't change, move onto next human.
            if human_idx in humans_to_keep:
                human_idx += 1
            # free humans if possible
            elif assignment_count >= min(self.exp_cfg.num_envs, len(env_priorities)):
                if self.assignments[:, human_idx].sum():
                    current_env = np.argmax(self.assignments[:, human_idx])
                    self.assignments[current_env][human_idx] = 0
                self.human_timers[human_idx] = 0
                human_idx += 1
            # Check if the next worst robot already has a human assigned.
            elif self.assignments[env_priorities[assignment_count]].sum():
                # If so, skip this robot.
                assignment_count += 1
            # If we have a human we can re-assign and a robot that doesn't already have a human, re-assign them
            else:
                if self.assignments[:, human_idx].sum():
                    current_env = np.argmax(self.assignments[:, human_idx])
                    self.assignments[current_env][human_idx] = 0
                new_env = env_priorities[assignment_count]
                self.assignments[new_env][human_idx] = 1
                self.human_timers[human_idx] = 0
                assignment_count += 1
                human_idx += 1
