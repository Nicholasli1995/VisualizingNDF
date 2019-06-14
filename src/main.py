import parse
import model
import dataset
import trainer
import optimizer

import logging
import torch

def main():
    # logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    
    # command line paser
    opt = parse.parse_arg()

    # GPU
    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
    else:
        logging.info("WARNING: RUN WITHOUT GPU")
    
    # prepare dataset    
    db = dataset.prepare_db(opt)
    
    # initalize neural decision forest
    NDF = model.prepare_model(opt)
    
    # prepare optimizer
    optim, sche = optimizer.prepare_optim(NDF, opt)
    
    # train the neural decision forest
    best_acc = trainer.train(NDF, optim, sche, db, opt)
    logging.info('The best evaluation accuracy is %f'%best_acc)

if __name__ == '__main__':
    main()

