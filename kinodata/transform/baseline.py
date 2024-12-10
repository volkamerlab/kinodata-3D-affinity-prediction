from torch_geometric.transforms import BaseTransform


class RemoveComponent(BaseTransform):
    component = None

    def __init__(self):
        pass

    def get_mask(self, data):
        raise NotImplementedError

    def __call__(self, data):
        mask = self.get_mask(data)
        x = data["complex"].x
        data["complex"].x = x[mask]
        return data


class RemovePocket(RemoveComponent):

    def get_mask(self, data):
        return data["complex"].is_pocket_atom


class RemoveLigand(RemoveComponent):

    def get_mask(self, data):
        return ~data["complex"].is_pocket_atom
