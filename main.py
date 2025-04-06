
import os

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckptdir", None, "Checkpoint directory.")
flags.DEFINE_enum("mode", None, ["train", 'deblur', "eval"],
                  "Running mode: train or eval or deblur")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_string("data_folder", None, "The folder name for training patches")
flags.mark_flags_as_required(["workdir", "config", "mode", "data_folder"])


def main(argv):
    if FLAGS.mode == "train":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline

    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
    elif FLAGS.mode == "deblur":
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
