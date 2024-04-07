## MaskNet-Occlusion-Invariant-Face-Recognition-System
MaskNet is an occlusion invariant face recognition solution built to overcome the limitations of recognizing faces while wearing a face mask. 

[What is MaskNet?](https://garnet-cardamom-4d4.notion.site/MASKNET-c4a6277ac1f84fcdb92cc784ecef08ee)

## Research Results
Please find the research results of MaskNet : Occlusion Invariant Face Recognition for masked faces in the following link. [Results of the research.](https://www.researchgate.net/project/Face-Mask-Invariant-Face-Recognition-with-Identity-Verification)

## Setting up the project 
Clone the repository from GitHub

	git clone https://github.com/susithrj/MaskNet-Occluded-face-Recognition-System.git

Install project requirements

	pip install -r requirements.txt

Run this in the root dir to launch the application.

	python Dashboard.py

This has a extensible, user operable facial recognition application.However, there are limitations with it as well. Anyone wishing to work on the limitations are welcome to do so.

## Technologies, Frameworks and Tools
    Python 3.6
    scikit-learn
    matplotlib
  opencv-contrib-python
    tensorflow== 2.4.1
    keras == 2.4.3
    scipy
    h5py

## Usage
## Contributing
## Credits
## cLicense

# MASKNET

Can Face recognition tech identify you while wearing a face mask?

!https://prod-files-secure.s3.us-west-2.amazonaws.com/3947affc-9d44-426c-8379-570927154d6c/42bd3b35-1afa-46de-a6f5-71be77bd6972/image4.png

Recognizing faces while wearing a face mask is considered the most difficult facial occlusion challenge because it occludes a large area that usually covers around 60% of the frontal face which contains rich features including the nose and mouth. This is emerging problem which higlighted during the pandemic period. I was able to develop a appropriate solution to overcome the above issue through a research study. The developed occlusion invariant face recognition system is published as a open work to the public.

[https://garnet-cardamom-4d4.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F3947affc-9d44-426c-8379-570927154d6c%2F42bd3b35-1afa-46de-a6f5-71be77bd6972%2Fimage4.png?table=block&id=6270c6b5-2013-4cb9-b455-d8671a10187f&spaceId=3947affc-9d44-426c-8379-570927154d6c&width=1400&userId=&cache=v2](https://file.notion.so/f/f/3947affc-9d44-426c-8379-570927154d6c/42bd3b35-1afa-46de-a6f5-71be77bd6972/image4.png?id=6270c6b5-2013-4cb9-b455-d8671a10187f&table=block&spaceId=3947affc-9d44-426c-8379-570927154d6c&expirationTimestamp=1712563200000&signature=qKq5fu-p7kpJJtpHqE-46z6XrS-NL6xwL-UCXRdI1Eg&downloadName=image4.png)

Rapid development and breakthrough of deep learning in the recent past have witnessed the most promising results from face recognition algorithms. But they fail to perform far from satisfactory levels in the unconstrained environment during the challenges such as varying lighting conditions, low resolution, facial expressions, pose variation and occlusions (Wen et al., 2016). Among the challenges, Occlusion is one of the most challenging problems faced by current face recognition algorithms. Moreover, Face mask makes higher inter-class similarity and inter-class variations due to covering a large area of the face which tricks the facial verification. process of face recognition systems (Guo and Zhang, 2019). Ongoing Face Recognition Vendor Test (FRVT) (Ngan, Grother and Hanaoka, 2020) is a performance evaluation report based on 89 face verification algorithms while wearing face masks (Ngan, Grother and Hanaoka, 2020). The study (Damer et al., 2020), concludes their results on the effect of wearing a face mask which given strong points of performance degradation on face recognition systems, showing the necessity of the development of mask capable face recognition systems.

!https://prod-files-secure.s3.us-west-2.amazonaws.com/3947affc-9d44-426c-8379-570927154d6c/b76ea02a-f79a-4382-b3d8-4acdac55c876/image5.png

Facial recognition has been studied for several years and it’s a longitudinal & preliminary task in computer vision (Best-Rowden and Jain, 2018). Among the biometrics, methodologies face recognition has a better edge on the accuracy, efficiency, usability, security, and privacy to recognize the identity non-intrusively. Furthermore, Facial recognition is widely preferred over potential candidates because of its contactless authentication behaviour which helps to keep good hygiene practices.(Zeng, Veldhuis and Spreeuwers, 2020). Face recognition is widely used in personal device log-on, passport checking, ATMs, border control, forensics, law enforcement, surveillance, and national security (Ejaz et al., 2019).

!https://prod-files-secure.s3.us-west-2.amazonaws.com/3947affc-9d44-426c-8379-570927154d6c/73bedd38-6f29-4d7a-ac69-19eae1bdd57b/image3.png

The study, by (Damer et al., 2020) concludestheir evaluationsidentified strong signs of an adverse effect of face masks on existing face recognition algorithms and occlusion invariant face recognition algorithms, pointing out the failure of them to identify faces while wearing a mask and the necessity of developing appropriate face recognition solutions. Furthermore, Ongoing Face Recognition Vendor Test (FRVT) is a performance evaluation report based on 89 face recognition algorithms on masked faces. If the mask covers around 70% of the occluded area most algorithm’s failure rate is increased by 20% to 50% according to the National Institute of Standards and Technology, United States (Ngan, Grother and Hanaoka, 2020).

!https://prod-files-secure.s3.us-west-2.amazonaws.com/3947affc-9d44-426c-8379-570927154d6c/be9aaa66-6f3b-4b6f-ada5-d823ce2035cd/image1.png

The recent surveys on the effect of face mask occlusion on face recognition precisely illustrate that existing face recognition systems and occlusion invariant face recognition systems are failed to handle face mask occlusions. However, most of the work done in small occlusions such as spectacles, sunglasses and partial captures commonly appears in in-the-wild capture conditions. In the existing research literature, there are few efforts are focused to handle face mask occlusions that are not met expected accuracy, efficiency, usability and security. This research is focused on building an effective novel facial recognition algorithm with the capability of recognizing faces while wearing a face mask in real-time to overtake barriers of occlusion invariant face recognition with growing incentives of wearing face masks without reducing well-known qualities of facial recognition.
