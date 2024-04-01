import torch


class Model(torch.nn.Module):
    """
    This class is a wrapper on the model object returned by cellulus.
    It updates the `forward` function and handles cases when the input raw
    image is not (S, C, (Z), Y, X) type.
    """

    def __init__(self, model, selected_axes):
        super().__init__()
        self.model = model
        self.selected_axes = selected_axes

    def forward(self, raw):
        if "s" in self.selected_axes and "c" in self.selected_axes:
            pass
        elif "s" in self.selected_axes and "c" not in self.selected_axes:

            raw = torch.unsqueeze(raw, 1)
        elif "s" not in self.selected_axes and "c" in self.selected_axes:
            pass
        elif "s" not in self.selected_axes and "c" not in self.selected_axes:
            raw = torch.unsqueeze(raw, 1)
        return self.model(raw)

    @staticmethod
    def select_and_add_coordinates(outputs, coordinates):
        selections = []
        # outputs.shape = (b, c, h, w) or (b, c, d, h, w)
        for output, coordinate in zip(outputs, coordinates):
            if output.ndim == 3:
                selection = output[:, coordinate[:, 1], coordinate[:, 0]]
            elif output.ndim == 4:
                selection = output[
                    :, coordinate[:, 2], coordinate[:, 1], coordinate[:, 0]
                ]
            selection = selection.transpose(1, 0)
            selection += coordinate
            selections.append(selection)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selections, dim=0)

    def set_infer(self, p_salt_pepper, num_infer_iterations, device):
        self.model.eval()
        self.model.set_infer(p_salt_pepper, num_infer_iterations, device)
