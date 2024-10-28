# Error Category Recognition in Procedural Videos using TimeChat

## Annotation JSONs

----

File: **step_annotations.json**

> Contains metadata for each recording, along with error category.

```json
{
    "1_7": {
          "recording_id": "1_7",
          "steps": [
                {
                  "step_id": 3,
                  "start_time": 7.072,
                  "end_time": 46.288,
                  "description": "Coat -Coat a 6-oz. ramekin cup with cooking spray",
                  "has_errors": false
                },
                ...
          ]
    },
    ...
}
```
File: **error_annotations.json**

> Contains metadata for each recording, including error category and error description

```json
[
    {
        "recording_id": "1_28",
        "activity_id": 1,
        "is_error": true,
        "step_annotations": [
            {
                "description": "Cut -Cut the English muffin into two pieces with a knife",
                "step_id": 12,
                "errors": [
                    {
                        "tag": "Preparation Error",
                        "description": "Cut the muffin into two pieces with hand"
                    }
                ],
                "start_time": 4.121,
                "end_time": 40.381,
                "modified_description": "Cut -Cut the English muffin into two pieces with hand"
            },
            ...
        ]
    },
    ...
]
```
File: **normal_videos.json**

> Contains metadata of a normal recording without any errors for each recipe type

```json
{
      "1_x": {
            "steps": [
                  {
                        "step_id": 3,
                        "start_time": 7.072,
                        "end_time": 46.288,
                        "description": "Coat -Coat a 6-oz. ramekin cup with cooking spray",
                        "has_errors": false
                  },
                  ...
            ]
      },
      ...
}
```

----
## Experiments

### Task Verification

This variant checks if all the steps defined for each recipe are performed in each recording. For this variant, the prompts follow the template 

> You are given a cooking video. Please watch the video and answer the following question: {Did, Is} the person 
> {perform, doing} the step **recipe step**? Return the answer in the format of Yes or No.

File: **questions.json**

> Contains questions engineered for task verification for each recipe

```json
{
      "1_x": {
            "questions": [
                  "Is the person cutting the English muffin into two pieces with a knife?",
                  "Is the person coating a 6-oz. ramekin cup with cooking spray?",
                  "Is the person pouring 1 egg into the ramekin cup?",
                  "Is the person microwaving the ramekin cup uncovered on high for 30 seconds?",
                  "Is the person stirring the ramekin cup?",
                  ...
            ]
      },
      ...
}
```
----

### Error Category Recognition

Each error category JSON file contains prompts engineered specifically for the category. The error categories are defined in the dataset. The prompts are engineered by considering a correct recipe recording for each recipe.

#### Missing Step
File: **missing_error.json**

> Contains prompts to recognise if any steps are missing in the procedure

**Template**
> You are given a cooking video. Please watch the video and answer the following question: {Did, Is} the person 
> {perform, doing} the step **recipe step**? Return the answer in the format of Yes or No.

```json
{
      "1_x": {
            "questions": [
                  "Is the person cutting the English muffin into two pieces with a knife?",
                  "Is the person coating a 6-oz. ramekin cup with cooking spray?",
                  "Is the person pouring 1 egg into the ramekin cup?",
                  "Is the person microwaving the ramekin cup uncovered on high for 30 seconds?",
                  "Is the person stirring the ramekin cup?",
                  ...
            ]
      },
      ...
}
```
---
#### Preparation Error
File: **preparation_error.json**

> Contains prompts engineered to recognize preparation errors. The JSON follows close-ended prompt formatting along with 
> correct options and follow-up questions to verify correct preparation method is followed.

**Template**
> You are given a cooking video. Please watch the video and answer the following question: {What, Which, ... } {tool, ingredient, ... } is used for **recipe step** to make **recipe**? 
> {Select, Choose, ... } one of the options: {option 1, option 2 ... }.


```json
{
      "21_x": {
            "questions": [
                  {
                        "q": "What cooking tool is being used in the step? Answer from the options: (a. blender b. whisk c. food processor d. none).",
                        "correctans": "blender",
                        "followup": [
                          "Does the person add baking powder to the blender?"
                        ]
                  },
                  ...
            ]
      },
      ...
}
```
---
#### Order Error

File: **order_error.json**

> Contains prompts engineered from task-graphs to verify correct recipe step ordering to cook. For each step, follow-up
> prompts verify if all necessary steps prior to current step are performed.

**Template**
> You are given a cooking video. Please watch the video and answer the following question: {Did, Is, Does, ... }the person 
> {perform, execute, doing, ... } the step **recipe step** to {cook, make} **recipe**? {Has, Have, ... } the 
> **previous recipe step(s)** been {completed, performed, ... } before **recipe step**? Return the answer in the format of Yes or No.

```json
{
      "21_x": {
            "questions": [
                  {
                        "q": "Is the blender blitzed for 20 seconds?",
                        "followup": [
                          "Does the person add 1/2 tsp baking powder to the blender?",
                          "Does the person add 1 banana to the blender?",
                          "Does the person add an egg to the blender?",
                          "Is 1 heaped tbsp of flour added to the blender?"
                        ]
                  },
                  ...
            ]
      },
      ...
}
```
----
#### Measurement Error
File: **measurement_error.json**

> Contains prompts to check if the appropriate measurements are being used.

**Template**
> You are given a cooking video. Please watch the video and answer the following question: To {complete, cook, ... } the recipe **recipe**, the person should {prepare, do, ... } the step **recipe step**.
>{Does, Did, ... } the person {measure, weigh} the {ingredient} accurately? Return the answer in the format of Yes or No.

```json
{
      "23_x": {
            "questions": [
                  {
                        "q": "Does the bowl have soy sauce added?",
                        "followup": [
                          "Is 1/8 cup of soy sauce being added?"
                        ]
                  },
                  ...
            ]
      },
      ...
}
```
----
#### Technique Error
File: **technique_error.json**

> Contains prompts to verify the correct technique is being used. The prompt includes the correct technique as reference

**Template**
> You are given a cooking video. Please watch the video and answer the following question:
> To prepare **recipe**, the person {should, has, ... } to {perform, execute, doing, ... } the step **recipe step**.
> {Did, Does, Has, ... } the person {carefully, precisely, ... } {perform, execute, doing, ... } the **recipe step**
> {without spilling, dropping, ... }? Return the answer in the format of Yes or No.

```json
{
      "1_x": {
            "questions": [
                  "To prepare the recipe, the person has to execute the step Coat -Coat a 6-oz. ramekin cup with cooking spray. Answer with a yes or no for the question: Did the person thoroughly coat the ramekin cup with cooking spray?",
                  "To prepare the recipe, the person has to execute the step Pour-Pour 1 egg into the ramekin cup. Answer with a yes or no for the question: Did the person carefully pour 1 egg without spilling into the ramekin cup?",
                  "To prepare the recipe, the person has to execute the step Microwave-Microwave the ramekin cup uncovered on high for 30 seconds. Answer with a yes or no for the question: Did the person microwave the ramekin cup uncovered on high for 30 seconds?",
                  ...
            ]
      },
      ...
}
```
----
#### Temperature Error
File: **temperature_error.json**

> Contains prompts to check if the appropriate temperature settings are followed for cooking the recipe.

**Template**
> You are given a cooking video. Please watch the video and answer the following question: {While, When ... } the person is {performing, executing, ... } 
> the step **recipe step** from the recipe **recipe**. 
> Is any heating involved? if yes, then did the person adhere to the {low, medium, high} {heating, power level} 
> settings of {microwave, stove}. Return the answer in the format of Yes or No.

```json
{
      "21_x": {
            "questions": [
                  {
                        "q": "To melt the butter, is heat required in the recipe?",
                        "followup": [
                          "Is the butter melted on low-medium heat?"
                        ]
                  },
                  ...
            ]
      },
      ...
}
```
----
#### Timing Error

File: **timing_error.json**

> Contains prompts to verify that the time for cooking is consistent with the recipe description.

**Template**
> You are given a cooking video. Please watch the video and answer the following question:
> To {cook, make, ... } **recipe**, the person {should, has, ... } to **recipe step**. {Should, Does}
> the person **recipe step** {perform, make, ... } for a {specific, certain} time? Return the answer in the format of Yes or No.

```json
{
      "21_x": {
            "questions": [
                  {
                        "q": "To perform this step, the person has to cook for 1 min or until the tops start to bubble. Should cooking be performed for a specific time?",
                        "followup": [
                          "Is cooking being performed for 1 min or until the tops start to bubble?"
                        ]
                  },
                  ...
            ]
      },
      ...
}
```

----

## Results

For task verification, the model is evaluated based on the ability to recognise step completion. \
For error category recognition, the model is evaluated on its ability to recognise category specific error. The combined
metrics for the data are calculated by combining each individual error category.
