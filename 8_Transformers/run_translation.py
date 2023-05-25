"""
Task 5 - Translation
Demonstration of full Transformer model
"""

import os
import numpy as np
import codecs
import regex
import json
from absl import flags
from absl import app
import seaborn
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lib.transformer import TransformerModel

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("plot", False, "Plot attention heatmaps")
flags.DEFINE_bool("cuda", False, "Use cuda")

# Training parameters
flags.DEFINE_integer("epochs", 1, "Number of training epochs")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_string(
    "savepath", os.path.join("data", "pretrained_model"), "Path to save or load model"
)
flags.DEFINE_integer("batchsize", 64, "Training batchsize per step")

# Model parameters
flags.DEFINE_integer("num_heads", 4, "Number of heads for multihead attention")
flags.DEFINE_integer("enc_layers", 1, "Number of self-attention layers for encodings")
flags.DEFINE_integer("dec_layers", 6, "Number of self-attention layers for encodings")
flags.DEFINE_integer("embed_dim", 64, "The dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 20, "Maximum input length from toy task")
flags.DEFINE_integer("line", None, "Line to test")


class Task(object):
    def __init__(self):
        base_path = os.path.join("data", "translation_data")
        self.en_file = os.path.join(base_path, "train.tags.de-en.en")
        self.de_file = os.path.join(base_path, "train.tags.de-en.de")
        self.en_samples = self.get_samples(self.en_file)
        self.de_samples = self.get_samples(self.de_file)
        self.rand_de = np.random.RandomState(1)
        self.rand_en = np.random.RandomState(1)
        self.n_samples = len(self.en_samples)
        with open(os.path.join(base_path, "en_dict.json"), "r", encoding="utf-8") as f:
            self.en_dict = json.load(f)
        with open(os.path.join(base_path, "de_dict.json"), "r", encoding="utf-8") as f:
            self.de_dict = json.load(f)
        self.en_vocab_size = len(self.en_dict)
        self.de_vocab_size = len(self.de_dict)
        self.idx = 0

    def get_samples(self, file):
        text = codecs.open(file, "r", "utf-8").read().lower()
        text = regex.sub(r"<.*>.*</.*>\r\n", "", text)
        text = regex.sub(r"[^\n\s\p{Latin}']", "", text)
        samples = text.split("\n")
        return samples

    def embed(self, sample, dictionary, max_len=20, sos=False, eos=False):
        sample = sample.split()[:max_len]
        while len(sample) < max_len:
            sample.append("<PAD>")
        if sos:
            tokens = ["<START>"]
        else:
            tokens = []
        tokens.extend(sample)
        if eos:
            tokens.append("<PAD>")
        idxs = []
        for token in tokens:
            try:
                idxs.append(dictionary.index(token))
            except ValueError:
                idxs.append(dictionary.index("<UNK>"))
        idxs = np.array(idxs)
        return np.eye(len(dictionary))[idxs]

    def next_batch(self, batchsize=64, max_len=20, idx=None):
        start = self.idx
        if idx is not None:
            start = idx
        end = start + batchsize
        if end > self.n_samples:
            end -= self.n_samples
            en_minibatch_text = self.en_samples[start:]
            self.rand_en.shuffle(self.en_samples)
            en_minibatch_text += self.en_samples[:end]
            de_minibatch_text = self.de_samples[start:]
            self.rand_de.shuffle(self.de_samples)
            de_minibatch_text += self.de_samples[:end]
        else:
            en_minibatch_text = self.en_samples[start:end]
            de_minibatch_text = self.de_samples[start:end]
        self.idx = end
        en_minibatch_in = []
        en_minibatch_out = []
        de_minibatch = []
        for sample in en_minibatch_text:
            en_minibatch_in.append(
                self.embed(sample, self.en_dict, max_len=max_len, sos=True)
            )
            en_minibatch_out.append(
                self.embed(sample, self.en_dict, max_len=max_len, eos=True)
            )
        for sample in de_minibatch_text:
            de_minibatch.append(self.embed(sample, self.de_dict, max_len=max_len))
        return (
            np.array(de_minibatch),
            np.array(en_minibatch_in),
            np.array(en_minibatch_out),
        )

    def prettify(self, sample, dictionary):
        idxs = np.argmax(sample, axis=1)
        return " ".join(np.array(dictionary)[idxs])


class TaskDataset(Dataset):
    def __init__(self, task, max_len):
        super().__init__()
        self.task = task
        self.max_len = max_len

    def __len__(self):
        return self.task.n_samples

    def __getitem__(self, i):
        ans = [
            x[0] for x in self.task.next_batch(batchsize=1, max_len=self.max_len, idx=i)
        ]
        return ans


def train(
    max_len=20,
    embed_dim=64,
    enc_layers=1,
    dec_layers=6,
    num_heads=4,
    batchsize=64,
    epochs=1,
    print_every=50,
    savepath=os.path.join("data", "pretrained_model"),
    cuda=torch.cuda.is_available(),
):

    os.makedirs(savepath, exist_ok=True)
    task = Task()
    device = torch.device("cuda:0" if cuda else "cpu")
    # print("Device: ", device)
    model = TransformerModel(
        task.en_vocab_size,
        task.de_vocab_size,
        max_len,
        embed_dim,
        enc_layers,
        dec_layers,
        num_heads,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True
    )
    task_dataset = TaskDataset(task, int(max_len))
    task_dataloader = DataLoader(
        task_dataset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=7
    )

    for i in range(epochs):
        print("Epoch: ", i)
        this_epoch_loss = 0
        for j, a_batch in enumerate(task_dataloader):
            minibatch_enc_in, minibatch_dec_in, minibatch_dec_out = a_batch
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                minibatch_enc_in = minibatch_enc_in.float().to(device)
                minibatch_dec_in = minibatch_dec_in.float().to(device)
                minibatch_dec_out = minibatch_dec_out.to(device)
                out, _ = model(minibatch_enc_in, minibatch_dec_in)
                loss = F.cross_entropy(
                    out.transpose(1, 2), minibatch_dec_out.argmax(dim=2)
                )
                loss.backward()
                optimizer.step()
            loss = loss.detach().cpu().numpy()
            this_epoch_loss += loss
            if (j + 1) % print_every == 0:
                print(f"Iteration {j + 1} - Loss {loss}")

        this_epoch_loss /= j + 1
        print(f"Epoch {i} - Loss {this_epoch_loss}")

        lr_scheduler.step(this_epoch_loss)

        torch.save(model.state_dict(), os.path.join(savepath, f"ckpt_{i}.pt"))
        print("Model saved")

    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(savepath, "ckpt_student.pt"))


def test(
    max_len=20,
    embed_dim=64,
    enc_layers=1,
    dec_layers=6,
    num_heads=4,
    savepath=os.path.join("data", "pretrained_model"),
    plot=True,
    line=198405,
    cuda=torch.cuda.is_available(),
):

    task = Task()
    device = torch.device("cuda:0" if cuda else "cpu")
    print("Device: ", device)
    model = TransformerModel(
        task.en_vocab_size,
        task.de_vocab_size,
        max_len,
        embed_dim,
        enc_layers,
        dec_layers,
        num_heads,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(savepath, "ckpt.pt"), device))

    idx = line
    if idx is None:
        idx = np.random.randint(low=0, high=task.n_samples)
        print("Predicting line :", idx)

    samples, _, truth = task.next_batch(batchsize=1, max_len=max_len, idx=idx)
    input_data = regex.sub(r"\s<PAD>", "", task.prettify(samples[0], task.de_dict))
    ground_truth = regex.sub(r"\s<PAD>", "", task.prettify(truth[0], task.en_dict))
    print(f"\nInput : \n{input_data}")
    print(f"\nTruth : \n{ground_truth}")

    output = ""
    for i in range(max_len):
        predictions, attention = model(
            torch.Tensor(samples).to(device),
            torch.Tensor(task.embed(output, task.en_dict, sos=True)).to(device),
        )
        predictions = predictions.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()
        output += " " + task.prettify(predictions[0], task.en_dict).split()[i]
    output = regex.sub(r"\s<PAD>", "", task.prettify(predictions[0], task.en_dict))
    print(f"\nOutput: \n{output}")

    if plot:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.5)
        x_labels = regex.sub(
            r"\s<PAD>", "", task.prettify(samples[0], task.de_dict)
        ).split()
        y_labels = regex.sub(
            r"\s<PAD>", "", task.prettify(predictions[0], task.en_dict)
        ).split()
        for i in range(4):
            ax = plt.Subplot(fig, gs[i])
            seaborn.heatmap(
                data=attention[:, 0, :, :][i, : len(y_labels), : len(x_labels)],
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax,
                cmap="plasma",
                vmin=np.min(attention),
                vmax=np.max(attention),
                cbar=False,
            )
            ax.set_title(f"Head {i}")
            ax.set_aspect("equal")
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            for tick in ax.get_yticklabels():
                tick.set_rotation(0)
            fig.add_subplot(ax)
        plt.show()


def main(unused_args):
    if FLAGS.train:
        train(
            FLAGS.max_len,
            FLAGS.embed_dim,
            FLAGS.enc_layers,
            FLAGS.dec_layers,
            FLAGS.num_heads,
            FLAGS.batchsize,
            FLAGS.epochs,
            FLAGS.print_every,
            FLAGS.savepath,
            FLAGS.cuda,
        )
    elif FLAGS.test:
        test(
            FLAGS.max_len,
            FLAGS.embed_dim,
            FLAGS.enc_layers,
            FLAGS.dec_layers,
            FLAGS.num_heads,
            FLAGS.savepath,
            FLAGS.plot,
            FLAGS.line,
            FLAGS.cuda,
        )
    else:
        print("Specify train or test option")


if __name__ == "__main__":
    app.run(main)
