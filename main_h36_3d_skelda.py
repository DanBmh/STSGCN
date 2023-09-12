import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils.h36_3d_viz import visualize
from utils.loss_funcs import *
from utils.parser import args

sys.path.append("/PoseForecasters/")
import utils_pipeline

# ==================================================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)


model = Model(
    args.input_dim,
    args.input_n,
    args.output_n,
    args.st_gcnn_dropout,
    args.joints_to_consider,
    args.n_tcnn_layers,
    args.tcnn_kernel_size,
    args.tcnn_dropout,
).to(device)
model_name = "h36_3d_" + str(args.output_n) + "frames_ckpt"

print(
    "total number of parameters of the network is: "
    + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
)

datapath_save_out = "/datasets/tmp/human36m/{}_forecast_samples.json"
config = {
    "item_step": 2,
    "window_step": 2,
    "input_n": 50,
    "output_n": 25,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        # "middlefoot_right",
        # "forefoot_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        # "middlefoot_left",
        # "forefoot_left",
        # "spine_upper",
        # "neck",
        "nose",
        # "head",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        # "hand_left",
        # "thumb_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        # "hand_right",
        # "thumb_right",
        "shoulder_middle",
    ],
}

viz_action = ""
# viz_action = "walking"

# ==================================================================================================


def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs) < 2:
        log_dir = os.path.join(out_dir, "exp0")
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, "exp%i" % (len(dirs) - 1))
        os.mkdir(log_dir)

    return log_dir


# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")
    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def train():

    log_dir = get_log_dir("./runs")
    tb_writer = SummaryWriter(log_dir=log_dir)
    print("Save data of the run in: %s" % log_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = utils_pipeline.load_dataset(
        datapath_save_out, "train", config
    )
    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        datapath_save_out, "eval", config
    )

    train_loss, val_loss, test_loss = [], [], []
    best_loss = np.inf

    for epoch in range(args.n_epochs):
        print("Run epoch: %i" % epoch)
        running_loss = 0
        model.train()

        label_gen_train = utils_pipeline.create_labels_generator(dataset_train, config)
        label_gen_eval = utils_pipeline.create_labels_generator(dataset_eval, config)

        nbatch = args.batch_size
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen_train, batch_size=nbatch),
            total=int(dlen_train / nbatch),
        ):

            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)
            optimizer.zero_grad()

            sequences_train = sequences_train.permute(0, 3, 1, 2)
            sequences_predict = model(sequences_train).permute(0, 1, 3, 2)

            loss = mpjpe_error(sequences_predict, sequences_gt)
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            running_loss += loss * nbatch
        train_loss.append(
            running_loss.detach().cpu() / (int(dlen_train / nbatch) * nbatch)
        )

        if args.use_scheduler:
            scheduler.step()

        eval_loss = run_eval(model, label_gen_eval, dlen_eval, args)
        val_loss.append(eval_loss)

        tb_writer.add_scalar("loss/train", train_loss[-1].item(), epoch)
        tb_writer.add_scalar("loss/val", val_loss[-1].item(), epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

        if eval_loss < best_loss:
            best_loss = eval_loss
            print("New best validation loss: %f" % best_loss)
            print("Saving best model...")
            if not os.path.isdir(args.model_path):
                os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_path, model_name))


# ==================================================================================================


def run_eval(model, dataset_gen_eval, dlen_eval, args):

    model.eval()

    with torch.no_grad():
        running_loss = 0

        nbatch = args.batch_size_test
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(dataset_gen_eval, batch_size=nbatch),
            total=int(dlen_eval / nbatch),
        ):

            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)

            sequences_train = sequences_train.permute(0, 3, 1, 2)
            sequences_predict = model(sequences_train).permute(0, 1, 3, 2)

            loss = mpjpe_error(sequences_predict, sequences_gt)
            running_loss += loss * nbatch

        avg_loss = running_loss.detach().cpu() / (int(dlen_eval / nbatch) * nbatch)
        print("overall average loss in mm is: {:.3f}".format(avg_loss))
        return avg_loss


# ==================================================================================================


def viz_joints_3d(sequences_predict, batch):
    batch = batch[0]
    vis_seq_pred = (
        sequences_predict.cpu()
        .detach()
        .numpy()
        .reshape(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
    )[0]
    utils_pipeline.visualize_pose_trajectories(
        np.array([cs["bodies3D"][0] for cs in batch["input"]]),
        np.array([cs["bodies3D"][0] for cs in batch["target"]]),
        utils_pipeline.make_absolute_with_last_input(vis_seq_pred, batch),
        batch["joints"],
        {"room_size": [3200, 4800, 2000], "room_center": [0, 0, 1000]},
    )
    plt.show()


# ==================================================================================================


def test():

    model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
    model.eval()

    # Load preprocessed datasets
    dataset_test, dlen = utils_pipeline.load_dataset(datapath_save_out, "test", config)
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, config)

    stime = time.time()
    frame_losses = np.zeros([args.output_n])
    nitems = 0

    with torch.no_grad():
        nbatch = 1

        for batch in tqdm.tqdm(label_gen_test, total=dlen):

            if nbatch == 1:
                batch = [batch]

            if viz_action != "" and viz_action != batch[0]["action"]:
                continue

            nitems += nbatch
            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)

            sequences_train = sequences_train.permute(0, 3, 1, 2)
            sequences_predict = model(sequences_train).permute(0, 1, 3, 2)

            loss = mpjpe_error(sequences_predict, sequences_gt)

            if viz_action != "":
                viz_joints_3d(sequences_predict, batch)

            loss = torch.sqrt(
                torch.sum(
                    (sequences_predict - sequences_gt) ** 2,
                    dim=-1,
                )
            )
            loss = torch.sum(torch.mean(loss, dim=2), dim=0)
            frame_losses += loss.cpu().data.numpy()

    avg_losses = frame_losses / nitems
    print("Averaged frame losses in mm are:", avg_losses)

    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))


# ==================================================================================================


if __name__ == "__main__":

    if args.mode == "train":
        stime = time.time()
        train()
        ftime = time.time()
        print("Training took {} seconds".format(int(ftime - stime)))
    elif args.mode == "test":
        test()
    elif args.mode == "viz":
        model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
        model.eval()
        visualize(
            args.input_n,
            args.output_n,
            args.visualize_from,
            args.data_dir,
            model,
            device,
            args.n_viz,
            args.skip_rate,
            args.actions_to_consider,
        )
