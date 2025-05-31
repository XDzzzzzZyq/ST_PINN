
import os

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["pretrain", "train", 'deblur', "eval"],
                  "Running mode: train or eval or deblur")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_string("data_folder", None, "The folder name for training patches")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):                         # tensorboard --logdir=workdir/large/tensorboard
    if FLAGS.mode == "train":           # python main.py --config config/default_configs.py --mode train --workdir workdir/adjoint
        from train import train         # python main.py --config config/simulate_configs.py --mode train --workdir workdir/simu
                                        # python main.py --config config/large_configs.py --mode train --workdir workdir/large
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
        train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "pretrain":
        from train import pretrain      # python main.py --config config/combine_configs.py --mode pretrain --workdir workdir/simu2
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
        pretrain(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval":
        # TODO: Evaluation pipeline
        pass
    elif FLAGS.mode == "deblur":        # python main.py --config config/large_configs.py --mode deblur --workdir workdir/large
        from deblur import deblur
        deblur(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
        pass
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
