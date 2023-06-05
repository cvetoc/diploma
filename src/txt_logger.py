import os

class TXTLogger:
    def __init__(self, work_dir, filename="progress_log.txt"):
        self.save_dir = work_dir
        self.filename = filename

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_file_path = os.path.join(self.save_dir, self.filename)
        log_file = open(self.log_file_path, 'w')
        log_file.close()

    def log(self, data):
        with open(self.log_file_path, 'a') as f:
            f.write(f'{str(data)}\n')