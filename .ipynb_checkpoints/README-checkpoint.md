# azure_ct

In this repository, I  have created a workflow to create a CI/CD pipeline for :
1. training a ML model
2.  register model
3. Depoly model using Azure Computer Instance way as rest api
4. score the model 
using GITHUB ACTIONS

--- steps in github actions workflow:
ON Event: 
    1. every time there is a PUSH to 'main' branch of the code, the github actions comes into action
Jobs Events:
    steps:
    2. runs on ubuntu
    3. checks out code from github
    4. setup python env
    5. install all the dependencies
    6. run training step: within training step we defined a Azure ML Pipeline for preparing the data , training the model,
       register model
    7.run deploying step: deploys model using ACI process
    8. Scoring file to score new data
