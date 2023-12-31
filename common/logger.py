import os
import atexit


class Logger:
    def __init__(self, writer, output_fname="progress.txt"):
        self.writer = writer
        self.log_path = self.writer.get_logdir()
        self.output_file = open(os.path.join(self.writer.get_logdir(), output_fname), 'w')
        atexit.register(self.output_file.close)
        self.record_file = open(os.path.join(self.writer.get_logdir(), "penalty.txt"), 'w')
        atexit.register(self.record_file.close)
        self.record_file = open(os.path.join(self.writer.get_logdir(), "mse.txt"), 'w')
        atexit.register(self.record_file.close)

    def record(self, tag, scalar_value, global_step, printed=True):
        self.writer.add_scalar(tag, scalar_value, global_step)
        if printed:
            info = f"{tag}: {scalar_value:.3f}"
            print("\033[1;32m [info]\033[0m: " + info)
            self.output_file.write(info + '\n')
    
    def print(self, info):
        print("\033[1;32m [info]\033[0m: " + info)
        self.output_file.write(info + '\n')

    def record_file(self, info, mse=True):
        if mse:
            count = 0
            self.record_file.write("mse epoch:" + count + info + '\n')
        else:
            count = 0
            self.record_file.write("mse epoch:" + count + info + '\n')
