import numpy as np
import tensorflow as tf


class PlantWrapper:
    def __init__(self, skeleton, muscle_type, timestep=0.01):
        # initialize time
        self.dt = timestep

        # initialize rigid body (the skeleton)
        self.Skeleton = skeleton
        self.Skeleton.build(timestep=self.dt)
        self.output_dim = self.Skeleton.output_dim
        self.proprioceptive_delay = self.Skeleton.proprioceptive_delay
        self.visual_delay = self.Skeleton.visual_delay

        # initialize muscle system
        self.Muscle = muscle_type
        self.MusclePaths = []  # a list of all the muscle paths
        self.n_muscles = 0
        self.input_dim = 0
        self.muscle_name = []
        self.muscle_state_dim = self.Muscle.state_dim
        self.geometry_state_dim = 2 + self.Skeleton.dof  # musculotendon length & velocity + as many moments as dofs
        self.path_fixation_body = np.empty((1, 1, 0)).astype('float32')
        self.path_coordinates = np.empty((1, self.Skeleton.space_dim, 0)).astype('float32')
        self.muscle = np.empty(0).astype('float32')
        self.tobuild__muscle = self.Muscle.to_build_dict
        self.muscle_transitions = None
        self.row_splits = None
        self.built = False
        self.init = False

    def build(self):
        self.Skeleton.build(timestep=self.dt)
        self.Muscle.build(**self.tobuild__muscle)

    def setattr(self, name: str, value):
        self.__setattr__(name, value)

    def add_muscle(self, path_fixation_body: list, path_coordinates: list, name='', **kwargs):
        path_fixation_body = np.array(path_fixation_body).astype('float32').reshape((1, 1, -1))
        n_points = path_fixation_body.size
        path_coordinates = np.array(path_coordinates).astype('float32').T[np.newaxis, :, :]
        assert path_coordinates.shape[1] == self.Skeleton.space_dim
        assert path_coordinates.shape[2] == n_points
        self.n_muscles += 1
        self.input_dim += self.Muscle.input_dim

        # path segments & coordinates should be a (batch_size * n_coordinates  * n_segments * (n_muscles * n_points)
        self.path_fixation_body = np.concatenate([self.path_fixation_body, path_fixation_body], axis=-1)
        self.path_coordinates = np.concatenate([self.path_coordinates, path_coordinates], axis=-1)
        self.muscle = np.hstack([self.muscle, np.tile(np.max(self.n_muscles), [n_points])])

        # indexes where the next item is from a different muscle, to indicate when their difference is meaningless
        self.muscle_transitions = np.diff(self.muscle.reshape((1, 1, -1))) == 1.
        # to create the ragged tensors when collapsing each muscle's segment values
        n_total_points = np.array([len(self.muscle)])
        self.row_splits = np.concatenate([np.zeros(1), np.diff(self.muscle).nonzero()[0] + 1, n_total_points - 1])

        kwargs.setdefault('timestep', self.dt)
        for key, val in kwargs.items():
            if key in self.tobuild__muscle:
                self.tobuild__muscle[key].append(val)
        for key, val in self.tobuild__muscle.items():
            if len(val) < self.n_muscles:
                raise ValueError('Missing keyword argument ' + key + '.')
        self.Muscle.build(**self.tobuild__muscle)

        if name == '':
            name = 'muscle_' + str(self.n_muscles)
        self.muscle_name.append(name)

    def get_initial_state(self, batch_size=1, skeleton_state=None):
        if skeleton_state is None:
            skeleton0 = self.Skeleton.draw_random_uniform_states(batch_size=batch_size)
        else:
            batch_size = skeleton_state.shape[0]
            skeleton0 = skeleton_state
        cartesian0 = self.Skeleton.joint2cartesian(joint_pos=skeleton0)
        geometry0 = self.Skeleton.get_geometry(path_coordinates=self.path_coordinates,
                                               path_fixation_body=self.path_fixation_body,
                                               muscle_transitions=self.muscle_transitions,
                                               row_splits=self.row_splits,
                                               skeleton_state=skeleton0)
        muscle0 = self.Muscle.get_initial_muscle_state(batch_size=batch_size, geometry=geometry0)
        state0 = [skeleton0, cartesian0, muscle0, geometry0]
        return state0

    # def draw_random_targets(self, batch_size=1, n_timesteps=1):
    #     targets = self.Skeleton.draw_random_uniform_positions(batch_size=batch_size)
    #     targets = targets[:, :, tf.newaxis]
    #     targets = tf.repeat(targets, n_timesteps, axis=-1)
    #     targets = tf.transpose(targets, [0, 2, 1])
    #     return targets

    def __call__(self, muscle_input, skeleton_state, muscle_state, geometry_state, **kwargs):
        endpoint_loads = kwargs.get('endpoint_loads', np.zeros(1))
        skeleton_loads = kwargs.get('skeleton_loads', np.zeros(1))
        skeleton_loads = tf.constant(skeleton_loads, shape=(1, self.Skeleton.dof), dtype=tf.float32)

        forces, new_muscle_state = self.Muscle(excitation=muscle_input,
                                               muscle_state=muscle_state,
                                               geometry_state=geometry_state)

        moments = tf.slice(geometry_state, [0, 2, 0], [-1, -1, -1])
        generalized_forces = - tf.reduce_sum(forces * moments, axis=-1) + skeleton_loads

        new_skeleton_state = self.Skeleton(generalized_forces, skeleton_state, endpoint_loads=endpoint_loads)
        new_cartesian_state = self.Skeleton.joint2cartesian(new_skeleton_state)
        new_geometry_state = self.Skeleton.get_geometry(path_coordinates=self.path_coordinates,
                                                        path_fixation_body=self.path_fixation_body,
                                                        muscle_transitions=self.muscle_transitions,
                                                        row_splits=self.row_splits,
                                                        skeleton_state=new_skeleton_state)

        return new_skeleton_state, new_cartesian_state, new_muscle_state, new_geometry_state
