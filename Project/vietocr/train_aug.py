from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer



def main():
    config = Cfg.load_config_from_name('vgg_transformer')


    dataset_params = {
        'name': 'BKAI',
        'data_root':'',
        'train_annotation':'/mlcv/WorkingSpace/SceneText/namnh/scene_text_pipeline/Data/data_crop/train/labels_full_original_augment.txt',
        'valid_annotation':'/mlcv/WorkingSpace/SceneText/namnh/scene_text_pipeline/Data/data_crop/valid/labels_full_original_augment.txt'
    }

    params = {
        'print_every':200,
        'valid_every':2000,
        'iters':20000,
        'batch_size': 64,
        ##'checkpoint':'./weights/vgg_transformer_BKAI_VINTEXT_2500_IMGS_AUG.pth',
        'export':'./weights/vgg_transformer_BKAI_AUG_1k5iter_add20kiter.pth',
        'metrics': 10000
    }
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:1'
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° '+ '̉'+ '̀' + '̃'+ '́'+ '̣'
    trainer = Trainer(config, pretrained=False)

    #trainer.config.save('my_config/vgg_transformer.yml')

    #trainer.load_checkpoint('./weights/vgg_transformer_BKAI_VINTEXT_2500_IMGS_AUG.pth')
    print('Weight loading')
    trainer.load_weights('./weights/vgg_transformer_BKAI_AUG_1k5iter.pth')
    print('Start training')
    trainer.train()

    #trainer.visualize_prediction()
    print(trainer.precision())

if __name__ == '__main__':
    main()
