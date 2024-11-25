from torch import nn
# define dice loss
class DiceBCELoss(nn.Module):
    """
    we expected inputs is probability value not logit
    """

    def __init__(self, eps=1e-7):
        super(DiceBCELoss, self).__init__()
        self.bceloss = nn.BCELoss()
        self.smooth = eps

    def forward(self, inputs, targets):
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate the Dice loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # Calculate the BCE loss
        bce_loss = self.bceloss(inputs, targets)

        # Combine the Dice loss and BCE loss
        dice_bce_loss = dice_loss + bce_loss
        # print(intersection, (inputs.sum() + targets.sum() + self.smooth), dice_bce_loss)

        return dice_bce_loss
    

if __name__ == '__main__':
    print("loss")