

class BaseSampler:
    @staticmethod
    def add_args(parser):
        """Add sampler-specific arguments to the parser."""
        pass

    @classmethod
    def build_sampler_from_args(cls, args):
        raise NotImplementedError("Samplers must implement the build_model_from_args")

    def __init__(self, loader):
        self.loader = loader
        self.num_batches = len(loader)
        self.iter = None

    def get_next_batch(self):
        raise NotImplementedError
    
    def reset_iter(self):
        self.iter = iter(self.loader)

    def get_loader(self):
        return self.loader

    def get_num_batches(self):
        return self.num_batches







    
    