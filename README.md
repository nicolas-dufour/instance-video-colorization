# Instance Video Colorization

This project aims to use the instance colorization model and apply it to video.

Video colorization has the constraint of temporal coherence. When we output the frames from the Image colorization model, we have indeed some color flickering. We could retrain the model on video data but we prefered to try to solve this problem without this.

We've reimplemented the architecture proposed in "Blind Video Temporal Consistency via Deep Video Prior" proposed by Lei et al. 

To see the image colorization process applied to video see this repo: https://github.com/liuvince/mva-video-colorization


| Black and White video             | Colored Frame by Frame                    | Colored and regularized                  |
| --------------------------------- | -----------------------------             | ---------------------------------------- |
|![](examples/swan-bw.gif)  | ![](examples/swan-colorized.gif)     | ![](examples/swan-smoothed.gif)   |




Instance colorization repo: https://github.com/ericsujw/InstColorization
