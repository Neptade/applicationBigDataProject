# application of Big Data Project 2

## How to run

- Clone the repository
- copy than paste the model to the model folder (we didn't upload it to github, it's too big)
- Open a CLI
- naviguate to the folder with the repository
- input command : docker compose up -d

The predictions will appear in a csv in the output folder. After initial an setup, only the 2 last steps are necessary.

### change prediction method

If you want to run the script that doesn't predict on images that have already been predicted :

- Open the dockerfile
- On the last line, change to second element of the list to : "classification_bonus.py"
- Push the change to Github
- Wait for a bit more than a minute
- Open a CLI
- naviguate to the folder with the repository
- input command : docker compose up -d

The predictions will appear in a csv in the output folder. 

## Description and implementation

The project is the creation of a docker container that applies a ResNet model to pictures to determine the weather. 
We created a docker image that continously uploads to Github Container Registry, on any git push. This way when you use the docker compose file to run the container you have the most up to date version. 
The custom image is based off an offical python image with the requirements installed and a prediction script primed to run at launch. 
Through the use of volumes you can interactively choose what data goes into the model, add more scripts into the app folder or use a different model. 