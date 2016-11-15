# Description
We are helping people with disorders/impairments by recognising, naming and interpreting their surroundings.

## Project Structure

### Model
For object classification, we're currently using Inception V3 model released by Google. This model is trained on ImageNet dataset, which has a lot of classes that we are not interested in (i.e exotic animals). We are mostly interested in household objects or objects we see in urban areas(TODO: what kind of classes are we interested in?). So we are looking into different datasets that cover the classes we're interested in as well as better models. 

### Application UI
We need more information to be able to do that.

## Competitors
### AIPoly
[AIPoly](http://www.aipoly.com/) is an iOS based application that helps blind and visually impaired people.
#### Features
1. Object Classification
2. Color Identification
  * Understands colors and assists blind and color blind people. 
3. Learn from descriptions
  * People can type descriptions to pictures and AIPoly can learn new object classes from this data.
4. Text Generation (in progress)
  * It can understand scenes and describe them in sentences. 

### Seeing AI
[Seeing AI](http://www.pivothead.com/seeingai/) is a collaboration between Microsoft and [Pivothead](http://www.pivothead.com)(a wearable smart glass startup). The [video](https://youtu.be/R2mC-NUAmMk) says it works on smart phones and pivothead smart glasses.
#### Features
1. Text Generation
2. Social Interaction
  * It can identify other people's age, gender and emotions. 
3. Read/classify documents
  * It can classify sections in a document and read them for the user. Show it a menu and ask it to read the headings.
4. Speech Input
  * It takes speech input, (read me the headings).
5. Speech Output
  * It speaks.

# Potential Ideas
### Pipeline
Creating a pipeline that optimizes the given model for mobile devices.

### Semantic Segmentation
Semantic Segmentation datasets contain interesting classes that could help us.

### Image to Text
Related datasets contain interesing classes. It may be possible to extract them from sentences.

### Object Retraining
At some point we may want users to take some pictures and train a new class based on this data.

### Depth
Knowing the distance of an object is important for visually impaired people.

