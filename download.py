# Runs during container build time to get model weights built into the container

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    try:
        model_id = "timbrooks/instruct-pix2pix"
        StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    except Exception as e:
        print("Failed dl: ", e)

    # def save_model_locally(self):
    #     # make dir
    #     self.model_tar_dir.mkdir()

    #     # setup temporary directory
    #     with TemporaryDirectory() as tmpdir:
    #         # download snapshot
    #         snapshot_dir = snapshot_download(repo_id=self.hf_model_id, cache_dir=tmpdir)
    #         # copy snapshot to model dir
    #         copy_tree(snapshot_dir, str(self.model_tar_dir))

    #     logger.info(f"Saved {self.hf_model_id} under {self.model_tar_dir}")
    


if __name__ == "__main__":
    download_model()