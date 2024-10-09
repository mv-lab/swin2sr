class MetricMonitor:
    def __init__(self, metric):
        assert metric in ['val_loss', 'val_psnr']
        self.metric = metric

    def mode(self):
        if self.metric == 'val_loss':
            return 'min'
        elif self.metric == 'val_psnr':
            return 'max'
        else:
            raise ValueError(f'Invalid metric:{self.metric}')
