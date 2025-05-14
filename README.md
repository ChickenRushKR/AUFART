# AUFART: Action Unit-Based 3D Face Reconstruction Using Transformers <br>
### Using Action Unit in single & monocular 3D Face reconstruction

![image](https://github.com/ChickenRushKR/AUFART/assets/67854851/b7513ea1-1bc8-404c-8e7d-ce2a3484eade)


This is the official Pytorch implementation of AUFART. <br>
The implementation is heavily dependent on DECA. <br>
AUFART reconstructs 3D head models sensitive to AU activations from a single monocular in-the-wild face image.<br>
Please refer to the paper 'A new method for 3D face reconstruction using transformers based on action unit features'.
[paper preprint]([https://assets-eu.researchsquare.com/files/rs-4310180/v1/26cf1775-0300-43db-ba2b-3b021cbc3c29.pdf?c=1714607300](https://www.sciencedirect.com/science/article/pii/S2405959525000499)) for more details.<br>

![image](https://github.com/ChickenRushKR/AUFART/assets/67854851/098ece63-f9be-44ac-bc81-4b18975610c5)

### Main contributions:<br>

* **AU-feature based reconstruction:** We propose a Transformer-based 3D face reconstruction framework that leverages the features of AUs in the frame-based 3D face reconstruction process, explicitly consider-ing their correlations.
* **Employing professional AU recognition model:** We integrate a state-of-the-art AU feature extraction module for effective AU feature extraction from in-the-wild images.
* **Effective architecture:** We propose a Transformer encoder-based model that can utilize both image features and AU features.
* **Novel loss functions:** We design AU-based loss fuctions for training our proposed 3D face reconstruction framework
  
## Related works:  
* for the accurate AU prediction: [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)  
* for the better skin estimation: [TRUST](https://github.com/HavenFeng/TRUST)
