import time

import torch
from PIL import Image
from torchvision import transforms

from app.infrastructure.ml.aligner.utils import model
from loguru import logger


class corner_finder:
    def __init__(self, CHECKPOINT_DIR):
        logger.info("Loading Corner Refiner model...")
        self.model = model.ModelFactory.get_model("resnet", "corner")
        model_data_dict = torch.load(CHECKPOINT_DIR, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model_state_dict = self.model.state_dict()
        missing_layers_keys = set([x for x in model_state_dict.keys()]) - set(
            [x for x in model_data_dict.keys()]
        )
        missing_layers = {x: model_state_dict[x] for x in missing_layers_keys}
        model_data_dict.update(missing_layers)
        self.model.load_state_dict(model_data_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        logger.info(f"Corner Refiner model loaded successfully on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Cache the transform
        self.test_transform = transforms.Compose(
            [transforms.Resize([32, 32]), transforms.ToTensor()]
        )
        
    def get_location(self, img, retainFactor=0.85):
        t_start = time.time()
        with torch.no_grad():
            ans_x = 0.0
            ans_y = 0.0

            # Avoid unnecessary copies
            o_img = img
            
            y = [0, 0]
            x_start = 0
            y_start = 0
            up_scale_factor = (img.shape[1], img.shape[0])

            myImage = o_img
            
            # Precompute the crop fraction to avoid repeated calculation
            CROP_FRAC = retainFactor
            
            # Track iterations for logging
            iterations = 0
            use_gpu = torch.cuda.is_available()
            
            # We'll stop after a certain number of iterations or when image is small enough
            max_iterations = 3  # Reduced for faster processing
            t_loop_start = time.time()
            
            while myImage.shape[0] > 10 and myImage.shape[1] > 10 and iterations < max_iterations:
                iterations += 1
                
                # Convert and transform image
                t_transform = time.time()
                img_temp = Image.fromarray(myImage)
                img_temp = self.test_transform(img_temp)
                img_temp = img_temp.unsqueeze(0)

                # Run model on GPU if available
                t_model = time.time()
                if use_gpu:
                    img_temp = img_temp.cuda(non_blocking=True)  # Asynchronous transfer
                    with torch.cuda.amp.autocast():  # Use mixed precision for faster computation
                        response = self.model(img_temp)
                    # Keep on GPU until necessary
                    response_cpu = response.cpu()
                    response = response_cpu.data.numpy()[0]
                else:
                    response = self.model(img_temp).data.numpy()[0]
                
                # Process response - GPU optimization opportunity
                t_process = time.time()
                
                # Scale response to match original image dimensions
                response_up = response * up_scale_factor
                y = response_up + (x_start, y_start)
                x_loc = int(y[0])
                y_loc = int(y[1])

                # Calculate crop bounds more efficiently
                height, width = myImage.shape[:2]
                half_crop_width = int(width * CROP_FRAC / 2)
                half_crop_height = int(height * CROP_FRAC / 2)
                crop_width = int(width * CROP_FRAC)
                crop_height = int(height * CROP_FRAC)
                
                # Determine crop start points
                if x_loc > width / 2:
                    start_x = min(x_loc + half_crop_width, width) - crop_width
                else:
                    start_x = max(x_loc - half_crop_width, 0)
                    
                if y_loc > height / 2:
                    start_y = min(y_loc + half_crop_height, height) - crop_height
                else:
                    start_y = max(y_loc - half_crop_height, 0)
                
                # Ensure crop bounds are valid
                start_x = max(0, min(start_x, width - 1))
                start_y = max(0, min(start_y, height - 1))
                crop_width = min(crop_width, width - start_x)
                crop_height = min(crop_height, height - start_y)
                
                # Update answer
                ans_x += start_x
                ans_y += start_y
                
                # Crop image more efficiently
                myImage = myImage[start_y:start_y + crop_height, start_x:start_x + crop_width]
                up_scale_factor = (myImage.shape[1], myImage.shape[0])
                
                # Early stopping if we're converging - more aggressive for speed
                if crop_width <= 20 or crop_height <= 20 or (iterations > 1 and (crop_width < width/2 or crop_height < height/2)):
                    break

            # Final position
            ans_x += y[0]
            ans_y += y[1]
            
            # Clear GPU cache if used
            if use_gpu:
                torch.cuda.empty_cache()
                
            return (int(round(ans_x)), int(round(ans_y)))


if __name__ == "__main__":
    pass
