class SAMCLIPArgs:
    part_batch_size: int = 32
    part_resolution: int = 224
    sam_size: int = 1024
    obj_feat_res: int = 100
    part_feat_res: int = 300
    final_feat_res: int = 64
    dino_resolution: int = 800
    dinov2_model_name: str = 'dinov2_vits14'
    mobilesamv2_encoder_name: str = 'mobilesamv2_efficientvit_l2'
    clip_model_name: str = 'ViT-L/14@336px'
    
    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "part_resolution": cls.part_resolution,
            "sam_size": cls.sam_size,
            "obj_feat_res": cls.obj_feat_res,
            "part_feat_res": cls.part_feat_res,
            "final_feat_res": cls.final_feat_res,
            "dino_resolution": cls.dino_resolution,
            "dinov2_model_name": cls.dinov2_model_name,
            "mobilesamv2_encoder_name": cls.mobilesamv2_encoder_name,
            "clip_model_name": cls.clip_model_name,
        }
