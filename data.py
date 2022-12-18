import jittor as jt
import jittor.dataset as dataset


class RayDataset(dataset.Dataset):
    def __init__(self, ray_data):
        super(RayDataset, self).__init__()

        self.rayData = ray_data
        self.length = ray_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return jt.array(self.rayData[index])


class RayPoseDataset(dataset.Dataset):
    def __init__(self, ray_data, ray_pose):
        super(RayPoseDataset, self).__init__()

        self.rayData = ray_data
        self.rayPose = ray_pose
        self.length = ray_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return jt.array(self.rayData[index]), jt.array(self.rayPose[index])
