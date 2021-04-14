import glob

if __name__ == '__main__':
    epcho_checkpoint_path = glob.glob("{}/model_epoch*".format("save_dir"))
    print(epcho_checkpoint_path)