# reflex_ai take home assignment
## Problem discussion
Download the AnnoMI dataset, the dataset includes transcripts from
  video demonstrations of high- and low-quality motivational interviewing. The dataset also
  includes annotations from experienced motivational interviewing practitioners.
  Perform analyses to explore 1 or 2 aspects of the dataset. 
  Then, create a text classification
  model that predicts the main_therapist_behaviour label. Evaluate the model that you created. In
  this hypothetical example, the model will be used to characterize the behavior of MI practitioners
  during their clinical sessions.

  ## Exploratory Data Analysis

In this anaylis I am using 'AnnoMI-simple.csv' as the AnnoMI-full.csv has duplicate value in the utterance_text column, diving deep into the data the droping duplicate may effect the analysis so for the time being I decided to do analysis on the simple file.
The destribution of the classes are shown below in the bar plot.

![image](https://github.com/SushaSureshh/reflex_ai/assets/35441892/01917d5f-20d3-40d4-908a-8188c51e32d9)

The data was split to test and train with at random with 10% for test and 90% for train. 

<img width="871" alt="Screenshot 2024-06-04 at 12 52 49 PM" src="https://github.com/SushaSureshh/reflex_ai/assets/35441892/cb03e763-2401-4f3c-80ff-472cfc822a6d">

<img width="846" alt="Screenshot 2024-06-04 at 12 53 13 PM" src="https://github.com/SushaSureshh/reflex_ai/assets/35441892/14050e49-5301-430f-b9bc-9974454e6704">

## Pre-processing
In the preprocessing step, I comnined the client utterance text to the therapist utterance text based on the time stap for context especially for the class Reflection and quetions, I thought that context is important.
1. The input sequence length of the utterance text - 
     - Min length: 3 tokens
     - Max length: 356 tokens
     - Avg length: 20.0 tokens
3. The occurnace of a unique sequence in the utternace_text column for the therapist only, top 5 are shown below.
utterance_text 


| utterance_text        | count           | 
| ------------- |:-------------:| 
|  Mm-hmm.     | 240 |
| Yeah.      | 182      |
| Okay. | 153    |
| Right.      | 75      |
| Mm. | 50     |




 






  
