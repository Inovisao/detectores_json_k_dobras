# Este dicionário abrange todas as redes disponíveis no mmdetections, vinculando-as às informações necessarias para realizar o treinamento da IA.

#   "config_file" -> refere-se ao nome do arquivo de configuração presente no mmdetection.
#   "checkpoint" -> indica o caminho para a pasta de checkpoints + o nome do arquivo .pth.
#   "model_download" -> link para o download do arquivo pth.

# É importante observar que o dicionário foi gerado automaticamente, e algumas redes deram problemas durante esse processo:
#   regnet
#   groie
#   cascade_rpn
#   ld
#   cornernet
#   tridentnet
#   carafe
#   legacy_1
#   fast_rcnn

# Caso queira utilizar alguma dessas redes, é necessário adicioná-las manualmente ao dicionário, fornecendo as informações necessárias.

import os

pasta_checkpoints=os.path.join(os.getcwd(),'checkpoints')

models_dict = {
    "sabl": {
        "sabl_faster_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth"
        },
        "sabl_faster_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/sabl/sabl_faster_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_faster_rcnn_r101_fpn_1x_coco-f804c6c1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/sabl_faster_rcnn_r101_fpn_1x_coco-f804c6c1.pth"
        },
        "sabl_cascade_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_cascade_rcnn_r50_fpn_1x_coco-e1748e5e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/sabl_cascade_rcnn_r50_fpn_1x_coco-e1748e5e.pth"
        },
        "sabl_cascade_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_cascade_rcnn_r101_fpn_1x_coco-2b83e87c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/sabl_cascade_rcnn_r101_fpn_1x_coco-2b83e87c.pth"
        },
        "sabl_retinanet_r50_fpn_1x_coco": {
            "config_file": "configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth"
        },
        "sabl_retinanet_r50_fpn_gn_1x_coco": {
            "config_file": "configs/sabl/sabl_retinanet_r50_fpn_gn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_retinanet_r50_fpn_gn_1x_coco-e16dfcf1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/sabl_retinanet_r50_fpn_gn_1x_coco-e16dfcf1.pth"
        },
        "sabl_retinanet_r101_fpn_1x_coco": {
            "config_file": "configs/sabl/sabl_retinanet_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_retinanet_r101_fpn_1x_coco-42026904.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/sabl_retinanet_r101_fpn_1x_coco-42026904.pth"
        },
        "sabl_retinanet_r101_fpn_gn_1x_coco": {
            "config_file": "configs/sabl/sabl_retinanet_r101_fpn_gn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_retinanet_r101_fpn_gn_1x_coco-40a893e8.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/sabl_retinanet_r101_fpn_gn_1x_coco-40a893e8.pth"
        },
        "sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco": {
            "config_file": "configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-1e63382c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-1e63382c.pth"
        },
        "sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco": {
            "config_file": "configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py",
            "checkpoint":  pasta_checkpoints + "/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-5342f857.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-5342f857.pth"
        }
    },
    "htc": {
        "htc_r50_fpn_1x_coco": {
            "config_file": "configs/htc/htc_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_r50_fpn_1x_coco_20200317-7332cf16.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth"
        },
        "htc_r50_fpn_20e_coco": {
            "config_file": "configs/htc/htc_r50_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_r50_fpn_20e_coco_20200319-fe28c577.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319-fe28c577.pth"
        },
        "htc_r101_fpn_20e_coco": {
            "config_file": "configs/htc/htc_r101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r101_fpn_20e_coco/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth"
        },
        "htc_x101_32x4d_fpn_16x1_20e_coco": {
            "config_file": "configs/htc/htc_x101_32x4d_fpn_16x1_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_x101_32x4d_fpn_16x1_20e_coco_20200318-de97ae01.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_32x4d_fpn_16x1_20e_coco/htc_x101_32x4d_fpn_16x1_20e_coco_20200318-de97ae01.pth"
        },
        "htc_x101_64x4d_fpn_16x1_20e_coco": {
            "config_file": "configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth"
        },
        "htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco": {
            "config_file": "configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth"
        }
    },
    "detr": {
        "detr_r50_8x2_150e_coco": {
            "config_file": "configs/detr/detr_r50_8x2_150e_coco.py",
            "checkpoint":  pasta_checkpoints + "/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"
        }
    },
    "cascade_rcnn": {
        "cascade_rcnn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth"
        },
        "cascade_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
        },
        "cascade_rcnn_r50_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"
        },
        "cascade_rcnn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth"
        },
        "cascade_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth"
        },
        "cascade_rcnn_r101_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth"
        },
        "cascade_rcnn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth"
        },
        "cascade_rcnn_x101_32x4d_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608-9ae0a720.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco/cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608-9ae0a720.pth"
        },
        "cascade_rcnn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth"
        },
        "cascade_rcnn_x101_64x4d_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth"
        },
        "cascade_mask_rcnn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.412__segm_mAP-0.36_20200504_174659-5004b251.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco/cascade_mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.412__segm_mAP-0.36_20200504_174659-5004b251.pth"
        },
        "cascade_mask_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
        },
        "cascade_mask_rcnn_r50_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth"
        },
        "cascade_mask_rcnn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.432__segm_mAP-0.376_20200504_174813-5c1e9599.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_caffe_fpn_1x_coco/cascade_mask_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.432__segm_mAP-0.376_20200504_174813-5c1e9599.pth"
        },
        "cascade_mask_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth"
        },
        "cascade_mask_rcnn_r101_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r101_fpn_20e_coco_bbox_mAP-0.434__segm_mAP-0.378_20200504_174836-005947da.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco/cascade_mask_rcnn_r101_fpn_20e_coco_bbox_mAP-0.434__segm_mAP-0.378_20200504_174836-005947da.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth"
        },
        "cascade_mask_rcnn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth"
        },
        "cascade_mask_rcnn_x101_64x4d_fpn_20e_coco": {
            "config_file": "configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth"
        }
    },
    "deformable_detr": {
        "deformable_detr_r50_16x2_50e_coco": {
            "config_file": "configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"
        },
        "deformable_detr_refine_r50_16x2_50e_coco": {
            "config_file": "configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth"
        },
        "deformable_detr_twostage_refine_r50_16x2_50e_coco": {
            "config_file": "configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth"
        }
    },
    "free_anchor": {
        "retinanet_free_anchor_r50_fpn_1x_coco": {
            "config_file": "configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth"
        },
        "retinanet_free_anchor_r101_fpn_1x_coco": {
            "config_file": "configs/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth"
        },
        "retinanet_free_anchor_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth"
        }
    },
    "gfl": {
        "gfl_r50_fpn_1x_coco": {
            "config_file": "configs/gfl/gfl_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth"
        },
        "gfl_r50_fpn_mstrain_2x_coco": {
            "config_file": "configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth"
        },
        "gfl_r101_fpn_mstrain_2x_coco": {
            "config_file": "configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth"
        },
        "gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco": {
            "config_file": "configs/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth"
        },
        "gfl_x101_32x4d_fpn_mstrain_2x_coco": {
            "config_file": "configs/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth"
        },
        "gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco": {
            "config_file": "configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth"
        }
    },
    "nas_fcos": {
        "nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco": {
            "config_file": "configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth"
        },
        "nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco": {
            "config_file": "configs/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521-7fdcbce0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521-7fdcbce0.pth"
        }
    },
    "mask_rcnn": {
        "mask_rcnn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth"
        },
        "mask_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
        },
        "mask_rcnn_r50_fpn_2x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
        },
        "mask_rcnn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth"
        },
        "mask_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth"
        },
        "mask_rcnn_r101_fpn_2x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_2x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth"
        },
        "mask_rcnn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth"
        },
        "mask_rcnn_x101_64x4d_fpn_2x_coco": {
            "config_file": "configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth"
        },
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        }
    },
    "deepfashion": {
        "mask_rcnn_r50_fpn_15e_deepfashion": {
            "config_file": "configs/deepfashion/mask_rcnn_r50_fpn_15e_deepfashion.py",
            "checkpoint":  pasta_checkpoints + "/open?id=1q6zF7J6Gb-FFgM87oIORIt6uBozaXp5r",
            "model_download": "https://drive.google.com/open?id=1q6zF7J6Gb-FFgM87oIORIt6uBozaXp5r"
        }
    },
    "retinanet": {
        "retinanet_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth"
        },
        "retinanet_r50_fpn_1x_coco": {
            "config_file": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
        },
        "retinanet_r50_fpn_2x_coco": {
            "config_file": "configs/retinanet/retinanet_r50_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
        },
        "retinanet_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/retinanet/retinanet_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth"
        },
        "retinanet_r101_fpn_1x_coco": {
            "config_file": "configs/retinanet/retinanet_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
        },
        "retinanet_r101_fpn_2x_coco": {
            "config_file": "configs/retinanet/retinanet_r101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth"
        },
        "retinanet_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth"
        },
        "retinanet_x101_32x4d_fpn_2x_coco": {
            "config_file": "configs/retinanet/retinanet_x101_32x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth"
        },
        "retinanet_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
        },
        "retinanet_x101_64x4d_fpn_2x_coco": {
            "config_file": "configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth"
        }
    },
    "vfnet": {
        "vfnet_r50_fpn_1x_coco": {
            "config_file": "configs/vfnet/vfnet_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth"
        },
        "vfnet_r50_fpn_mstrain_2x_coco": {
            "config_file": "configs/vfnet/vfnet_r50_fpn_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth"
        },
        "vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco": {
            "config_file": "configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth"
        },
        "vfnet_r101_fpn_1x_coco": {
            "config_file": "configs/vfnet/vfnet_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth"
        },
        "vfnet_r101_fpn_mstrain_2x_coco": {
            "config_file": "configs/vfnet/vfnet_r101_fpn_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth"
        },
        "vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco": {
            "config_file": "configs/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth"
        },
        "vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco": {
            "config_file": "configs/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth"
        },
        "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco": {
            "config_file": "configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth"
        }
    },
    "point_rend": {
        "point_rend_r50_caffe_fpn_mstrain_1x_coco": {
            "config_file": "configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth"
        },
        "point_rend_r50_caffe_fpn_mstrain_3x_coco": {
            "config_file": "configs/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth"
        }
    },
    "yolact": {
        "yolact_r50_1x8_coco": {
            "config_file": "configs/yolact/yolact_r50_1x8_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolact_r50_1x8_coco_20200908-f38d58df.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco_20200908-f38d58df.pth"
        },
        "yolact_r50_8x8_coco": {
            "config_file": "configs/yolact/yolact_r50_8x8_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolact_r50_8x8_coco_20200908-ca34f5db.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r50_8x8_coco_20200908-ca34f5db.pth"
        },
        "yolact_r101_1x8_coco": {
            "config_file": "configs/yolact/yolact_r101_1x8_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolact_r101_1x8_coco_20200908-4cbe9101.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco_20200908-4cbe9101.pth"
        }
    },
    "scnet": {
        "scnet_r50_fpn_1x_coco": {
            "config_file": "configs/scnet/scnet_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/view?usp=sharing",
            "model_download": "https://drive.google.com/file/d/1K5_8-P0EC43WZFtoO3q9_JE-df8pEc7J/view?usp=sharing"
        },
        "scnet_r50_fpn_20e_coco": {
            "config_file": "configs/scnet/scnet_r50_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/view?usp=sharing",
            "model_download": "https://drive.google.com/file/d/15VGLCt5-IO5TbzB4Kw6ZyoF6QH0Q511A/view?usp=sharing"
        },
        "scnet_r101_fpn_20e_coco": {
            "config_file": "configs/scnet/scnet_r101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/view?usp=sharing",
            "model_download": "https://drive.google.com/file/d/1aeCGHsOBdfIqVBnBPp0JUE_RSIau3583/view?usp=sharing"
        },
        "scnet_x101_64x4d_fpn_20e_coco": {
            "config_file": "configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/view?usp=sharing",
            "model_download": "https://drive.google.com/file/d/1YjgutUKz4TTPpqSWGKUTkZJ8_X-kyCfY/view?usp=sharing"
        }
    },
    "centripetalnet": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/centripetalnet_hourglass104_mstest_16x6_210e_coco.py",
            "model_download": "https://github.com/open-mmlab/mmdetection/tree/master/configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py"
        }
    },
    "paa": {
        "paa_r50_fpn_1x_coco": {
            "config_file": "configs/paa/paa_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r50_fpn_1x_coco_20200821-936edec3.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth"
        },
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "paa_r50_fpn_1.5x_coco": {
            "config_file": "configs/paa/paa_r50_fpn_1.5x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r50_fpn_1.5x_coco_20200823-805d6078.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.pth"
        },
        "paa_r50_fpn_2x_coco": {
            "config_file": "configs/paa/paa_r50_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r50_fpn_2x_coco_20200821-c98bfc4e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.pth"
        },
        "paa_r50_fpn_mstrain_3x_coco": {
            "config_file": "configs/paa/paa_r50_fpn_mstrain_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth"
        },
        "paa_r101_fpn_1x_coco": {
            "config_file": "configs/paa/paa_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth"
        },
        "paa_r101_fpn_2x_coco": {
            "config_file": "configs/paa/paa_r101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r101_fpn_2x_coco_20200821-6829f96b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.pth"
        },
        "paa_r101_fpn_mstrain_3x_coco": {
            "config_file": "configs/paa/paa_r101_fpn_mstrain_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth"
        }
    },
    "scratch": {
        "faster_rcnn_r50_fpn_gn-all_scratch_6x_coco": {
            "config_file": "configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py",
            "checkpoint":  pasta_checkpoints + "/scratch_faster_rcnn_r50_fpn_gn_6x_bbox_mAP-0.407_20200201_193013-90813d01.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_bbox_mAP-0.407_20200201_193013-90813d01.pth"
        },
        "mask_rcnn_r50_fpn_gn-all_scratch_6x_coco": {
            "config_file": "configs/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco.py",
            "checkpoint":  pasta_checkpoints + "/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth"
        }
    },
    "ms_rcnn": {
        "ms_rcnn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"
        },
        "ms_rcnn_r50_caffe_fpn_2x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth"
        },
        "ms_rcnn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.404__segm_mAP-0.376_20200506_004755-b9b12a37.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.404__segm_mAP-0.376_20200506_004755-b9b12a37.pth"
        },
        "ms_rcnn_r101_caffe_fpn_2x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth"
        },
        "ms_rcnn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth"
        },
        "ms_rcnn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth"
        },
        "ms_rcnn_x101_64x4d_fpn_2x_coco": {
            "config_file": "configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth"
        }
    },
    "dynamic_rcnn": {
        "dynamic_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/dynamic_rcnn_r50_fpn_1x-62a3f276.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x-62a3f276.pth"
        }
    },
    "fsaf": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200715_094657.log.json",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200715_094657.log.json"
        },
        "fsaf_r50_fpn_1x_coco": {
            "config_file": "configs/fsaf/fsaf_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fsaf_r50_fpn_1x_coco-94ccc51f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth"
        },
        "fsaf_r101_fpn_1x_coco": {
            "config_file": "configs/fsaf/fsaf_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fsaf_r101_fpn_1x_coco-9e71098f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco-9e71098f.pth"
        },
        "fsaf_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth"
        }
    },
    "gn": {
        "mask_rcnn_r50_fpn_gn-all_2x_coco": {
            "config_file": "configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth"
        },
        "mask_rcnn_r50_fpn_gn-all_3x_coco": {
            "config_file": "configs/gn/mask_rcnn_r50_fpn_gn-all_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214-8b23b1e5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214-8b23b1e5.pth"
        },
        "mask_rcnn_r101_fpn_gn-all_2x_coco": {
            "config_file": "configs/gn/mask_rcnn_r101_fpn_gn-all_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205-d96b1b50.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205-d96b1b50.pth"
        },
        "mask_rcnn_r101_fpn_gn-all_3x_coco": {
            "config_file": "configs/gn/mask_rcnn_r101_fpn_gn-all_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth"
        },
        "mask_rcnn_r50_fpn_gn-all_contrib_2x_coco": {
            "config_file": "configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207-20d3e849.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207-20d3e849.pth"
        },
        "mask_rcnn_r50_fpn_gn-all_contrib_3x_coco": {
            "config_file": "configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225-542aefbc.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225-542aefbc.pth"
        }
    },
    "double_heads": {
        "dh_faster_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth"
        }
    },
    "hrnet": {
        "faster_rcnn_hrnetv2p_w18_1x_coco": {
            "config_file": "configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth"
        },
        "faster_rcnn_hrnetv2p_w18_2x_coco": {
            "config_file": "configs/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731-a4ec0611.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731-a4ec0611.pth"
        },
        "faster_rcnn_hrnetv2p_w32_1x_coco": {
            "config_file": "configs/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_hrnetv2p_w32_1x_coco_20200130-6e286425.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130-6e286425.pth"
        },
        "faster_rcnn_hrnetv2p_w32_2x_coco": {
            "config_file": "configs/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth"
        },
        "faster_rcnn_hrnetv2p_w40_1x_coco": {
            "config_file": "configs/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_hrnetv2p_w40_1x_coco_20200210-95c1f5ce.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210-95c1f5ce.pth"
        },
        "faster_rcnn_hrnetv2p_w40_2x_coco": {
            "config_file": "configs/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth"
        },
        "mask_rcnn_hrnetv2p_w18_1x_coco": {
            "config_file": "configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_hrnetv2p_w18_1x_coco_20200205-1c3d78ed.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205-1c3d78ed.pth"
        },
        "mask_rcnn_hrnetv2p_w18_2x_coco": {
            "config_file": "configs/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_hrnetv2p_w18_2x_coco_20200212-b3c825b1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212-b3c825b1.pth"
        },
        "mask_rcnn_hrnetv2p_w32_1x_coco": {
            "config_file": "configs/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_hrnetv2p_w32_1x_coco_20200207-b29f616e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207-b29f616e.pth"
        },
        "mask_rcnn_hrnetv2p_w32_2x_coco": {
            "config_file": "configs/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_hrnetv2p_w32_2x_coco_20200213-45b75b4d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213-45b75b4d.pth"
        },
        "mask_rcnn_hrnetv2p_w40_1x_coco": {
            "config_file": "configs/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646-66738b35.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646-66738b35.pth"
        },
        "mask_rcnn_hrnetv2p_w40_2x_coco": {
            "config_file": "configs/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732-aed5e4ab.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732-aed5e4ab.pth"
        },
        "cascade_rcnn_hrnetv2p_w18_20e_coco": {
            "config_file": "configs/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth"
        },
        "cascade_rcnn_hrnetv2p_w32_20e_coco": {
            "config_file": "configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth"
        },
        "cascade_rcnn_hrnetv2p_w40_20e_coco": {
            "config_file": "configs/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112-75e47b04.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112-75e47b04.pth"
        },
        "cascade_mask_rcnn_hrnetv2p_w18_20e_coco": {
            "config_file": "configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210-b543cd2b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210-b543cd2b.pth"
        },
        "cascade_mask_rcnn_hrnetv2p_w32_20e_coco": {
            "config_file": "configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043-39d9cf7b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043-39d9cf7b.pth"
        },
        "cascade_mask_rcnn_hrnetv2p_w40_20e_coco": {
            "config_file": "configs/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922-969c4610.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922-969c4610.pth"
        },
        "htc_hrnetv2p_w18_20e_coco": {
            "config_file": "configs/hrnet/htc_hrnetv2p_w18_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_hrnetv2p_w18_20e_coco_20200210-b266988c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210-b266988c.pth"
        },
        "htc_hrnetv2p_w32_20e_coco": {
            "config_file": "configs/hrnet/htc_hrnetv2p_w32_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_hrnetv2p_w32_20e_coco_20200207-7639fa12.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207-7639fa12.pth"
        },
        "htc_hrnetv2p_w40_20e_coco": {
            "config_file": "configs/hrnet/htc_hrnetv2p_w40_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth"
        },
        "fcos_hrnetv2p_w18_gn-head_4x4_1x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth"
        },
        "fcos_hrnetv2p_w18_gn-head_4x4_2x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110-5c575fa5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110-5c575fa5.pth"
        },
        "fcos_hrnetv2p_w32_gn-head_4x4_1x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730-cb8055c0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730-cb8055c0.pth"
        },
        "fcos_hrnetv2p_w32_gn-head_4x4_2x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133-77b6b9bb.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133-77b6b9bb.pth"
        },
        "fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651-441e9d9f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651-441e9d9f.pth"
        },
        "fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846-b6f2b49f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846-b6f2b49f.pth"
        },
        "fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco": {
            "config_file": "configs/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth"
        }
    },
    "pascal_voc": {
        "faster_rcnn_r50_fpn_1x_voc0712": {
            "config_file": "configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth"
        },
        "retinanet_r50_fpn_1x_voc0712": {
            "config_file": "configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth"
        }
    },
    "legacy_1.x": {},
    "fcos": {
        "fcos_r50_caffe_fpn_gn-head_1x_coco": {
            "config_file": "configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"
        },
        "fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco": {
            "config_file": "configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth"
        },
        "fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco": {
            "config_file": "configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth"
        },
        "fcos_r101_caffe_fpn_gn-head_1x_coco": {
            "config_file": "configs/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth"
        },
        "fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco": {
            "config_file": "configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth"
        },
        "fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco": {
            "config_file": "configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth"
        },
        "fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco": {
            "config_file": "configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth",
            "model_download": "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth"
        }
    },
    "gcnet": {
        "mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915-187da160.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915-187da160.pth"
        },
        "mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth"
        },
        "mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco_20200205-e58ae947.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco_20200205-e58ae947.pth"
        },
        "mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco_20200206-af22dc9d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco_20200206-af22dc9d.pth"
        },
        "mask_rcnn_r50_fpn_syncbn-backbone_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202-bb3eb55c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202-bb3eb55c.pth"
        },
        "mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth"
        },
        "mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth"
        },
        "mask_rcnn_r101_fpn_syncbn-backbone_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco_20200210-81658c8a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco_20200210-81658c8a.pth"
        },
        "mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200207-945e77ca.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200207-945e77ca.pth"
        },
        "mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200211-7584841c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200211-7584841c.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-cbed3d2c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-cbed3d2c.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200212-68164964.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200212-68164964.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco": {
            "config_file": "configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200310-d5ad2a5e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200310-d5ad2a5e.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-10bf2463.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-10bf2463.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200703_180653-ed035291.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200703_180653-ed035291.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco_20200516_182249-680fc3f2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco_20200516_182249-680fc3f2.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco_20200516_015634-08f56b56.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco_20200516_015634-08f56b56.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco": {
            "config_file": "configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20200518_041145-24cabcfd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20200518_041145-24cabcfd.pth"
        }
    },
    "sparse_rcnn": {
        "sparse_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth"
        },
        "sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco": {
            "config_file": "configs/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth"
        },
        "sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco": {
            "config_file": "configs/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth"
        },
        "sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco": {
            "config_file": "configs/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth"
        },
        "sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco": {
            "config_file": "configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py",
            "checkpoint":  pasta_checkpoints + "/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth"
        }
    },
    "reppoints": {
        "bbox_r50_grid_fpn_gn-neck+head_1x_coco": {
            "config_file": "configs/reppoints/bbox_r50_grid_fpn_gn-neck+head_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329-c98bfa96.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329-c98bfa96.pth"
        },
        "bbox_r50_grid_center_fpn_gn-neck+Bhead_1x_coco": {
            "config_file": "configs/reppoints/bbox_r50_grid_center_fpn_gn-neck+Bhead_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco_20200330-00f73d58.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco_20200330-00f73d58.pth"
        },
        "reppoints_moment_r50_fpn_1x_coco": {
            "config_file": "configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth"
        },
        "reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco": {
            "config_file": "configs/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329-4b38409a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329-4b38409a.pth"
        },
        "reppoints_moment_r50_fpn_gn-neck+head_2x_coco": {
            "config_file": "configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329-91babaa2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329-91babaa2.pth"
        },
        "reppoints_moment_r101_fpn_gn-neck+head_2x_coco": {
            "config_file": "configs/reppoints/reppoints_moment_r101_fpn_gn-neck+head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329-4fbc7310.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329-4fbc7310.pth"
        },
        "reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck+head_2x_coco": {
            "config_file": "configs/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-3309fbf2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-3309fbf2.pth"
        },
        "reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco": {
            "config_file": "configs/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth"
        }
    },
    "cityscapes": {
        "faster_rcnn_r50_fpn_1x_cityscapes": {
            "config_file": "configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
        },
        "mask_rcnn_r50_fpn_1x_cityscapes": {
            "config_file": "configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth"
        }
    },
    "yolof": {
        "yolof_r50_c5_8x8_1x_coco": {
            "config_file": "configs/yolof/yolof_r50_c5_8x8_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth"
        }
    },
    "res2net": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "faster_rcnn_r2_101_fpn_2x_coco": {
            "config_file": "configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth"
        },
        "mask_rcnn_r2_101_fpn_2x_coco": {
            "config_file": "configs/res2net/mask_rcnn_r2_101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r2_101_fpn_2x_coco-17f061e8.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/res2net/mask_rcnn_r2_101_fpn_2x_coco/mask_rcnn_r2_101_fpn_2x_coco-17f061e8.pth"
        },
        "cascade_rcnn_r2_101_fpn_20e_coco": {
            "config_file": "configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth"
        },
        "cascade_mask_rcnn_r2_101_fpn_20e_coco": {
            "config_file": "configs/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth"
        },
        "htc_r2_101_fpn_20e_coco": {
            "config_file": "configs/res2net/htc_r2_101_fpn_20e_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_r2_101_fpn_20e_coco-3a8d2112.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/res2net/htc_r2_101_fpn_20e_coco/htc_r2_101_fpn_20e_coco-3a8d2112.pth"
        }
    },
    "regnet": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py",
            "model_download": "https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py"
        }
    },
    "faster_rcnn": {
        "faster_rcnn_r50_caffe_dc5_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909-531f0f43.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909-531f0f43.pth"
        },
        "faster_rcnn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth"
        },
        "faster_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_bounded_iou_1x_coco-98ad993b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_bounded_iou_1x_coco-98ad993b.pth"
        },
        "faster_rcnn_r50_fpn_2x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        },
        "faster_rcnn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth"
        },
        "faster_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth"
        },
        "faster_rcnn_r101_fpn_2x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth"
        },
        "faster_rcnn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth"
        },
        "faster_rcnn_x101_32x4d_fpn_2x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth"
        },
        "faster_rcnn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth"
        },
        "faster_rcnn_x101_64x4d_fpn_2x_coco": {
            "config_file": "configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth"
        },
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py",
            "model_download": "./faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py"
        }
    },
    "albu_example": {
        "mask_rcnn_r50_fpn_albu_1x_coco": {
            "config_file": "configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_albu_1x_coco_20200208-ab203bcd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/albu_example/mask_rcnn_r50_fpn_albu_1x_coco/mask_rcnn_r50_fpn_albu_1x_coco_20200208-ab203bcd.pth"
        }
    },
    "groie": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth"
        }
    },
    "dcn": {
        "faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth"
        },
        "faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130-d099253b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130-d099253b.pth"
        },
        "faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth"
        },
        "faster_rcnn_r50_fpn_dpool_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_dpool_1x_coco_20200307-90d3c01d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307-90d3c01d.pth"
        },
        "faster_rcnn_r50_fpn_mdpool_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth"
        },
        "faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth"
        },
        "faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth"
        },
        "mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth"
        },
        "mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-ad97591f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-ad97591f.pth"
        },
        "mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth"
        },
        "cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth"
        },
        "cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth"
        },
        "cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth"
        },
        "cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204-df0c5f10.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204-df0c5f10.pth"
        },
        "cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco": {
            "config_file": "configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth"
        }
    },
    "cascade_rpn": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        }
    },
    "empirical_attention": {
        "faster_rcnn_r50_fpn_attention_1111_1x_coco": {
            "config_file": "configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth"
        },
        "faster_rcnn_r50_fpn_attention_0010_1x_coco": {
            "config_file": "configs/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130-7cb0c14d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130-7cb0c14d.pth"
        },
        "faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco": {
            "config_file": "configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130-8b2523a6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130-8b2523a6.pth"
        },
        "faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco": {
            "config_file": "configs/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth"
        }
    },
    "pisa": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "pisa_faster_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth"
        },
        "pisa_faster_rcnn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth"
        },
        "pisa_mask_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_mask_rcnn_r50_fpn_1x_coco-dfcedba6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco-dfcedba6.pth"
        },
        "pisa_retinanet_r50_fpn_1x_coco": {
            "config_file": "configs/pisa/pisa_retinanet_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_retinanet_r50_fpn_1x_coco-76409952.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco-76409952.pth"
        },
        "pisa_retinanet_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_retinanet_x101_32x4d_fpn_1x_coco-a0c13c73.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco-a0c13c73.pth"
        },
        "pisa_ssd300_coco": {
            "config_file": "configs/pisa/pisa_ssd300_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_ssd300_coco-710e3ac9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco-710e3ac9.pth"
        },
        "pisa_ssd512_coco": {
            "config_file": "configs/pisa/pisa_ssd512_coco.py",
            "checkpoint":  pasta_checkpoints + "/pisa_ssd512_coco-247addee.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco-247addee.pth"
        }
    },
    "carafe": {
        "faster_rcnn_r50_fpn_carafe_1x_coco": {
            "config_file": "configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth"
        },
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "mask_rcnn_r50_fpn_carafe_1x_coco": {
            "config_file": "configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth"
        }
    },
    "yolo": {
        "yolov3_d53_320_273e_coco": {
            "config_file": "configs/yolo/yolov3_d53_320_273e_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolov3_d53_320_273e_coco-421362b6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth"
        },
        "yolov3_d53_mstrain-416_273e_coco": {
            "config_file": "configs/yolo/yolov3_d53_mstrain-416_273e_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth"
        },
        "yolov3_d53_mstrain-608_273e_coco": {
            "config_file": "configs/yolo/yolov3_d53_mstrain-608_273e_coco.py",
            "checkpoint":  pasta_checkpoints + "/yolov3_d53_mstrain-608_273e_coco-139f5633.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth"
        }
    },
    "autoassign": {
        "autoassign_r50_fpn_8x2_1x_coco": {
            "config_file": "configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth"
        }
    },
    "ld": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "ld_r18_gflv1_r101_fpn_coco_1x": {
            "config_file": "configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "ld_r34_gflv1_r101_fpn_coco_1x": {
            "config_file": "configs/ld/ld_r34_gflv1_r101_fpn_coco_1x.py",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "ld_r50_gflv1_r101_fpn_coco_1x": {
            "config_file": "configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "ld_r101_gflv1_r101dcn_fpn_coco_1x": {
            "config_file": "configs/ld/ld_r101_gflv1_r101dcn_fpn_coco_1x.py",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        }
    },
    "cornernet": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/cornernet_hourglass104_mstest_32x3_210e_coco.py",
            "model_download": "https://github.com/open-mmlab/mmdetection/tree/master/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py"
        }
    },
    "instaboost": {
        "mask_rcnn_r50_fpn_instaboost_4x_coco": {
            "config_file": "configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth"
        },
        "mask_rcnn_r101_fpn_instaboost_4x_coco": {
            "config_file": "configs/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth"
        },
        "mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco": {
            "config_file": "configs/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth"
        },
        "cascade_mask_rcnn_r50_fpn_instaboost_4x_coco": {
            "config_file": "configs/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-c19d98d9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-c19d98d9.pth"
        }
    },
    "resnest": {
        "faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco": {
            "config_file": "configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth"
        },
        "faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco": {
            "config_file": "configs/resnest/faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth"
        },
        "mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco": {
            "config_file": "configs/resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20200926_125503-8a2c3d47.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20200926_125503-8a2c3d47.pth"
        },
        "mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco": {
            "config_file": "configs/resnest/mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_215831-af60cdf9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_215831-af60cdf9.pth"
        },
        "cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco": {
            "config_file": "configs/resnest/cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201122_213640-763cc7b5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201122_213640-763cc7b5.pth"
        },
        "cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco": {
            "config_file": "configs/resnest/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth"
        },
        "cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco": {
            "config_file": "configs/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201122_104428-99eca4c7.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201122_104428-99eca4c7.pth"
        },
        "cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco": {
            "config_file": "configs/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_113243-42607475.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_113243-42607475.pth"
        }
    },
    "fast_rcnn": {},
    "nas_fpn": {
        "retinanet_r50_fpn_crop640_50e_coco": {
            "config_file": "configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth"
        },
        "retinanet_r50_nasfpn_crop640_50e_coco": {
            "config_file": "configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth"
        }
    },
    "libra_rcnn": {
        "libra_faster_rcnn_r50_fpn_1x_coco": {
            "config_file": "configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth"
        },
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        },
        "libra_faster_rcnn_r101_fpn_1x_coco": {
            "config_file": "configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth"
        },
        "libra_faster_rcnn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth"
        },
        "libra_retinanet_r50_fpn_1x_coco": {
            "config_file": "configs/libra_rcnn/libra_retinanet_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth"
        }
    },
    "ssd": {
        "ssd300_coco": {
            "config_file": "configs/ssd/ssd300_coco.py",
            "checkpoint":  pasta_checkpoints + "/ssd300_coco_20200307-a92d2092.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth"
        },
        "ssd512_coco": {
            "config_file": "configs/ssd/ssd512_coco.py",
            "checkpoint":  pasta_checkpoints + "/ssd512_coco_20200308-038c5591.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth"
        }
    },
    "tridentnet": {
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539.log.json",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539.log.json"
        }
    },
    "foveabox": {
        "fovea_r50_fpn_4x4_1x_coco": {
            "config_file": "configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth"
        },
        "fovea_r50_fpn_4x4_2x_coco": {
            "config_file": "configs/foveabox/fovea_r50_fpn_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth"
        },
        "fovea_align_r50_fpn_gn-head_4x4_2x_coco": {
            "config_file": "configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth"
        },
        "fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco": {
            "config_file": "configs/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth"
        },
        "fovea_r101_fpn_4x4_1x_coco": {
            "config_file": "configs/foveabox/fovea_r101_fpn_4x4_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth"
        },
        "fovea_r101_fpn_4x4_2x_coco": {
            "config_file": "configs/foveabox/fovea_r101_fpn_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_r101_fpn_4x4_2x_coco_20200208-02320ea4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208-02320ea4.pth"
        },
        "fovea_align_r101_fpn_gn-head_4x4_2x_coco": {
            "config_file": "configs/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208-c39a027a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208-c39a027a.pth"
        },
        "fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco": {
            "config_file": "configs/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth"
        }
    },
    "rpn": {
        "rpn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/rpn/rpn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_r50_caffe_fpn_1x_coco_20200531-5b903a37.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_caffe_fpn_1x_coco/rpn_r50_caffe_fpn_1x_coco_20200531-5b903a37.pth"
        },
        "rpn_r50_fpn_1x_coco": {
            "config_file": "configs/rpn/rpn_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth"
        },
        "rpn_r50_fpn_2x_coco": {
            "config_file": "configs/rpn/rpn_r50_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth"
        },
        "rpn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/rpn/rpn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_r101_caffe_fpn_1x_coco_20200531-0629a2e2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_caffe_fpn_1x_coco/rpn_r101_caffe_fpn_1x_coco_20200531-0629a2e2.pth"
        },
        "rpn_r101_fpn_1x_coco": {
            "config_file": "configs/rpn/rpn_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth"
        },
        "rpn_r101_fpn_2x_coco": {
            "config_file": "configs/rpn/rpn_r101_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth"
        },
        "rpn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/rpn/rpn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth"
        },
        "rpn_x101_32x4d_fpn_2x_coco": {
            "config_file": "configs/rpn/rpn_x101_32x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth"
        },
        "rpn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/rpn/rpn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth"
        },
        "rpn_x101_64x4d_fpn_2x_coco": {
            "config_file": "configs/rpn/rpn_x101_64x4d_fpn_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth"
        }
    },
    "guided_anchoring": {
        "ga_rpn_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_rpn_r50_caffe_fpn_1x_coco_20200531-899008a6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco/ga_rpn_r50_caffe_fpn_1x_coco_20200531-899008a6.pth"
        },
        "ga_rpn_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_rpn_r101_caffe_fpn_1x_coco_20200531-ca9ba8fb.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco/ga_rpn_r101_caffe_fpn_1x_coco_20200531-ca9ba8fb.pth"
        },
        "ga_rpn_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth"
        },
        "ga_rpn_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_rpn_x101_64x4d_fpn_1x_coco_20200225-3c6e1aa2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225-3c6e1aa2.pth"
        },
        "ga_faster_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth"
        },
        "ga_faster_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_faster_r101_caffe_fpn_1x_coco_bbox_mAP-0.415_20200505_115528-fb82e499.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco/ga_faster_r101_caffe_fpn_1x_coco_bbox_mAP-0.415_20200505_115528-fb82e499.pth"
        },
        "ga_faster_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_faster_x101_32x4d_fpn_1x_coco_20200215-1ded9da3.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco/ga_faster_x101_32x4d_fpn_1x_coco_20200215-1ded9da3.pth"
        },
        "ga_faster_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_faster_x101_64x4d_fpn_1x_coco_20200215-0fa7bde7.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco/ga_faster_x101_64x4d_fpn_1x_coco_20200215-0fa7bde7.pth"
        },
        "ga_retinanet_r50_caffe_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth",
            "model_download": "https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth"
        },
        "ga_retinanet_r101_caffe_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_retinanet_r101_caffe_fpn_1x_coco_20200531-6266453c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco/ga_retinanet_r101_caffe_fpn_1x_coco_20200531-6266453c.pth"
        },
        "ga_retinanet_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth"
        },
        "ga_retinanet_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth"
        },
        "": {
            "config_file": "",
            "checkpoint":  pasta_checkpoints + "/",
            "model_download": ""
        }
    },
    "atss": {
        "atss_r50_fpn_1x_coco": {
            "config_file": "configs/atss/atss_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth"
        },
        "atss_r101_fpn_1x_coco": {
            "config_file": "configs/atss/atss_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/atss_r101_fpn_1x_20200825-dfcadd6f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r101_fpn_1x_coco/atss_r101_fpn_1x_20200825-dfcadd6f.pth"
        }
    },
    "lvis": {
        "mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5": {
            "config_file": "configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth"
        },
        "mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_v0.5": {
            "config_file": "configs/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis-54582ee2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis-54582ee2.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5": {
            "config_file": "configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis-3cf55ea2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis-3cf55ea2.pth"
        },
        "mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5": {
            "config_file": "configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis-1c99a5ad.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis-1c99a5ad.pth"
        },
        "mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1": {
            "config_file": "configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth"
        },
        "mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1": {
            "config_file": "configs/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1": {
            "config_file": "configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth"
        },
        "mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1": {
            "config_file": "configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth"
        }
    },
    "detectors": {
        "cascade_rcnn_r50_rfp_1x_coco": {
            "config_file": "configs/detectors/cascade_rcnn_r50_rfp_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth"
        },
        "cascade_rcnn_r50_sac_1x_coco": {
            "config_file": "configs/detectors/cascade_rcnn_r50_sac_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/cascade_rcnn_r50_sac_1x_coco-24bfda62.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco-24bfda62.pth"
        },
        "detectors_cascade_rcnn_r50_1x_coco": {
            "config_file": "configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth"
        },
        "htc_r50_rfp_1x_coco": {
            "config_file": "configs/detectors/htc_r50_rfp_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_r50_rfp_1x_coco-8ff87c51.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco-8ff87c51.pth"
        },
        "htc_r50_sac_1x_coco": {
            "config_file": "configs/detectors/htc_r50_sac_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/htc_r50_sac_1x_coco-bfa60c54.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco-bfa60c54.pth"
        },
        "detectors_htc_r50_1x_coco": {
            "config_file": "configs/detectors/detectors_htc_r50_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/detectors_htc_r50_1x_coco-329b1453.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth"
        }
    },
    "pafpn": {
        "faster_rcnn_r50_pafpn_1x_coco": {
            "config_file": "configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth"
        }
    },
    "grid_rcnn": {
        "grid_rcnn_r50_fpn_gn-head_2x_coco": {
            "config_file": "configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth"
        },
        "grid_rcnn_r101_fpn_gn-head_2x_coco": {
            "config_file": "configs/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309-d6eca030.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309-d6eca030.pth"
        },
        "grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco": {
            "config_file": "configs/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130-d8f0e3ff.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130-d8f0e3ff.pth"
        },
        "grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco": {
            "config_file": "configs/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth"
        }
    },
    "fpg": {
        "faster_rcnn_r50_fpg_crop640_50e_coco": {
            "config_file": "configs/fpg/faster_rcnn_r50_fpg_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpg_crop640_50e_coco-76220505.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/faster_rcnn_r50_fpg_crop640_50e_coco-76220505.pth"
        },
        "faster_rcnn_r50_fpg-chn128_crop640_50e_coco": {
            "config_file": "configs/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpg-chn128_crop640_50e_coco-24257de9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/faster_rcnn_r50_fpg-chn128_crop640_50e_coco-24257de9.pth"
        },
        "mask_rcnn_r50_fpg_crop640_50e_coco": {
            "config_file": "configs/fpg/mask_rcnn_r50_fpg_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpg_crop640_50e_coco-c5860453.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/mask_rcnn_r50_fpg_crop640_50e_coco-c5860453.pth"
        },
        "mask_rcnn_r50_fpg-chn128_crop640_50e_coco": {
            "config_file": "configs/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpg-chn128_crop640_50e_coco-5c6ea10d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/mask_rcnn_r50_fpg-chn128_crop640_50e_coco-5c6ea10d.pth"
        },
        "retinanet_r50_fpg_crop640_50e_coco": {
            "config_file": "configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpg_crop640_50e_coco-46fdd1c6.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/retinanet_r50_fpg_crop640_50e_coco-46fdd1c6.pth"
        },
        "retinanet_r50_fpg-chn128_crop640_50e_coco": {
            "config_file": "configs/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpg-chn128_crop640_50e_coco-5cf33c76.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/retinanet_r50_fpg-chn128_crop640_50e_coco-5cf33c76.pth"
        }
    },
    "ghm": {
        "retinanet_ghm_r50_fpn_1x_coco": {
            "config_file": "configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth"
        },
        "retinanet_ghm_r101_fpn_1x_coco": {
            "config_file": "configs/ghm/retinanet_ghm_r101_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth"
        },
        "retinanet_ghm_x101_32x4d_fpn_1x_coco": {
            "config_file": "configs/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth"
        },
        "retinanet_ghm_x101_64x4d_fpn_1x_coco": {
            "config_file": "configs/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth"
        }
    },
    "fp16": {
        "faster_rcnn_r50_fpn_fp16_1x_coco": {
            "config_file": "configs/fp16/faster_rcnn_r50_fpn_fp16_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth"
        },
        "mask_rcnn_r50_fpn_fp16_1x_coco": {
            "config_file": "configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_fp16_1x_coco_20200205-59faf7e4.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_1x_coco/mask_rcnn_r50_fpn_fp16_1x_coco_20200205-59faf7e4.pth"
        },
        "retinanet_r50_fpn_fp16_1x_coco": {
            "config_file": "configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth"
        }
    },
    "gn+ws": {
        "faster_rcnn_r50_fpn_gn_ws-all_1x_coco": {
            "config_file": "configs/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth"
        },
        "faster_rcnn_r101_fpn_gn_ws-all_1x_coco": {
            "config_file": "configs/gn%2Bws/faster_rcnn_r101_fpn_gn_ws-all_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_r101_fpn_gn_ws-all_1x_coco_20200205-a93b0d75.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r101_fpn_gn_ws-all_1x_coco/faster_rcnn_r101_fpn_gn_ws-all_1x_coco_20200205-a93b0d75.pth"
        },
        "faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco": {
            "config_file": "configs/gn%2Bws/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco_20200203-839c5d9d.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco_20200203-839c5d9d.pth"
        },
        "faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco": {
            "config_file": "configs/gn%2Bws/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco.py",
            "checkpoint":  pasta_checkpoints + "/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco_20200212-27da1bc2.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco_20200212-27da1bc2.pth"
        },
        "mask_rcnn_r50_fpn_gn_ws-all_2x_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_gn_ws-all_2x_coco_20200226-16acb762.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco/mask_rcnn_r50_fpn_gn_ws-all_2x_coco_20200226-16acb762.pth"
        },
        "mask_rcnn_r101_fpn_gn_ws-all_2x_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_gn_ws-all_2x_coco_20200212-ea357cd9.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_2x_coco/mask_rcnn_r101_fpn_gn_ws-all_2x_coco_20200212-ea357cd9.pth"
        },
        "mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco_20200319-33fb95b5.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco_20200319-33fb95b5.pth"
        },
        "mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco_20200213-487d1283.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco_20200213-487d1283.pth"
        },
        "mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213-57b5a50f.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213-57b5a50f.pth"
        },
        "mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200226-969bcb2c.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200226-969bcb2c.pth"
        },
        "mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco": {
            "config_file": "configs/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco.py",
            "checkpoint":  pasta_checkpoints + "/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200316-e6cd35ef.pth",
            "model_download": "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200316-e6cd35ef.pth"
        }
    }
}