'''
VISA_TO_MVTEC = {'seen': ['candle'],
                 'unseen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']}
'''
VISA_TO_MVTEC = {'seen': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
                 'unseen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']}

MVTEC_TO_VISA = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']}

MVTEC_TO_BTAD = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                          'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                          'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['01', '02', '03']}

MVTEC_TO_MVTEC3D = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                             'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                             'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                            'foam', 'peach', 'potato', 'rope', 'tire']}

MVTEC_TO_MPDD = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                             'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                             'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['bracket_black', 'bracket_brown', 'bracket_white',
                            'connector', 'metal_plate', 'tubes']}

MVTEC_TO_MVTECLOCO = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                             'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                             'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag',
                            'splicing_connectors']}

MVTEC_TO_BRATS = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                             'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                             'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['brain']}

MVTEC_TO_KSDD = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                             'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                             'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['KSDD']}

MVTEC_TO_ReinADtest = {'seen': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                             'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                             'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                 'unseen': ['PCB_solder_1', 'PCB_solder_2', 'bearing_1', 'cable_1', 'cable_3', 'led_1', 'led_3', 'lens_2', 'motor_base_1', 'motor_base_2', 'motor_base_4', 'motor_base_5', 'motor_base_7', 'motor_base_8', 'piston_ring_1', 'piston_ring_3', 'plastic_box', 'plastic_cover_1', 'profile_surface_1', 'thread', 'wafer_1', 'wafer_2']}

ReinADtrain_TO_ReinADtest = {'seen': ['PCB_terminal', 'bearing_10', 'bearing_11', 'bearing_12', 'bearing_2', 'bearing_3', 'bearing_4', 'bearing_5', 'bearing_6', 'bearing_7', 'bearing_8', 'bearing_9', 'cable_2', 'led_2', 'lens_1', 'miniled_1', 'miniled_2', 'miniled_3', 'motor_base_10', 'motor_base_11', 'motor_base_3', 'motor_base_6', 'motor_base_9', 'pinline', 'piston_ring_2', 'plastic_cover_2', 'plastic_cover_3', 'plastic_cover_4', 'plastic_cover_5', 'plastic_cover_6', 'profile_surface_2', 'reflective_sheet', 'round_tube', 'solder_1', 'solder_2', 'solder_3', 'suspension_wire'],
                 'unseen': ['PCB_solder_1', 'PCB_solder_2', 'bearing_1', 'cable_1', 'cable_3', 'led_1', 'led_3', 'lens_2', 'motor_base_1', 'motor_base_2', 'motor_base_4', 'motor_base_5', 'motor_base_7', 'motor_base_8', 'piston_ring_1', 'piston_ring_3', 'plastic_box', 'plastic_cover_1', 'profile_surface_1', 'thread', 'wafer_1', 'wafer_2']}
