//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include "cnn_unet.h"
//#include "tensor.h"
//#include "utils.h"
//
//// stb_image for loading input images
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
//// stb_image_write for saving output masks
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
//
//int main_infer(int argc, char** argv) {
//    if (argc != 4) {
//        fprintf(stderr, "Usage: %s <model.bin> <input.png> <output_mask.png>\n", argv[0]);
//        return 1;
//    }
//    const char* model_path = argv[1];
//    const char* input_path = argv[2];
//    const char* output_path = argv[3];
//
//    // 1) Build U-Net and load trained weights
//    UNetModel model;
//    unet_build(&model);
//    if (!load_model(&model, model_path)) {
//        fprintf(stderr, "Error: failed to load model from '%s'\n", model_path);
//        return 1;
//    }
//
//    // 2) Load input image (grayscale)
//    int W, H, C;
//    unsigned char* img_data = stbi_load(input_path, &W, &H, &C, 1);
//    if (!img_data) {
//        fprintf(stderr, "Error: could not load image '%s'\n", input_path);
//        return 1;
//    }
//
//    // 3) Create input tensor [1,1,H,W], normalize to [0,1]
//    int shape[4] = { 1, 1, H, W };
//    Tensor* input = create_tensor(shape, 4);
//    for (int i = 0, n = H * W; i < n; ++i) {
//        input->values[i] = img_data[i] / 255.0f;
//    }
//    stbi_image_free(img_data);
//
//    // 4) Forward pass
//    UNetIntermediates* im = unet_forward(&model, input);
//    Tensor* pred = im->pred_mask; // shape [1,1,H,W]
//
//    // 5) Post-process: threshold at 0.5
//    unsigned char* mask = (unsigned char*)malloc(H * W);
//    for (int i = 0, n = H * W; i < n; ++i) {
//        float v = pred->values[i];
//        mask[i] = (v >= 0.5f) ? 255 : 0;
//    }
//
//    // 6) Save output mask
//    if (!stbi_write_png(output_path, W, H, 1, mask, W)) {
//        fprintf(stderr, "Error: failed to write mask to '%s'\n", output_path);
//    }
//    else {
//        printf("Saved segmentation mask to '%s'\n", output_path);
//    }
//
//    // 7) Cleanup
//    free(mask);
//    free_tensor(input);
//    unet_free_intermediates(im);
//    unet_free(&model);
//
//    return 0;
//}
