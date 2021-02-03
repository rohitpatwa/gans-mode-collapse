# Hyper paramertes

class Arguments:
    def __init__(self):

        self.dataset = 'mnist'
        self.dataroot = '../data'
        self.workers = 2
        self.batchSize = 64 
        self.imageSize = 64
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.niter = 50
        self.lr = 0.0002
        self.beta1 = 0.5
        self.cuda = True
        self.ngpu = 1
        self.netG = ''
        self.netD = ''
        self.outf = '../models'
        self.manualSeed = None
        self.error_percentage = 0.05

opt = Arguments()
print(opt)