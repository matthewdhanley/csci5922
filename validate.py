import torch
import time
from utils.metrics import runningScore


def validate(model, dataloader, checkpoint_path, save_path=None):
    since = time.time()

    # set the device to gpu if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Testing Device: {}\n".format(device))
    model.to(device)
    metrics = runningScore(model.num_classes)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.cpu().numpy()
        metrics.update(gt, pred)

    time_elapsed = time.time() - since

    score, class_iou = metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(model.num_classes):
        print(i, class_iou[i])

    return model

