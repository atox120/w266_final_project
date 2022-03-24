# w266_final_project - Cloud Access

Access via the ssh key is in a terminal. Note this assumes that the .key file is in the working directory. 

```bash
ssh -i <ssh-key-file>.key ubuntu@130.162.193.21
```

Don't forget to have the read permissions on the key set via:

```bash
chmod 400 <ssh-key-file>.key
```

## Usage

### Setting up your workspace

First you need to clone this repository to get you our working files.
```bash
cd ~/<your_username>
git clone git@github.com:atox120/w266_final_project.git
```
Note be careful configuruing this directory. Make sure you don't set global gituhb configurations for your username and email when setting up the credentials for git. I have added an ssh key for this instance to the repository, but I think that is only for my directory. Not sure if you need to do this seperately each. 

### Setting up your directory

The `src/source_chkpts` folder contains all model checkpoints. IMPORTANT: copy any files needed in your container into your local working folder BEFORE you run your container. We will map the container volume to your local directory /<your_username>/w266_final_project , which because it is at the same level as /src, you won't be able to see /src when inside the container, only the contents of your /<your_username> folder.
  
I reccomend running a command like:
  
```bash
cd ~
mkdir ~/<your_username>/w266_final_project/source
cp -r ~/src/* ~/<your_username>/w266_final_project/source
```

### Running a Jupyter Notebook:

So now you should have:

1. A clone of the repository in your folder ~/<your_username>
2. A copy of the ~/src folder in your clone of the repository, ~/<your_username>/w266_final_project/source

Next we can run the container image, noting that each of us needs to use a different port on the host. I've assigned ports as follows:

Port designation:

- Wanyu: 8886
- Rathin: 8882
- Alex: 8888

The container is prebuilt using the dockerfile in src/Dockerfile. The image name is `atox120/w266_fp_torch:v2`. To run the container please use the following command:

```bash
sudo docker run -it --ipc=host --rm -v /home/ubuntu/<your_username>/w266_final_project/:/workspace/w266_final_project -p <your_designated_port>:8888 --gpus=all atox120/w266_fp_torch:v2
```
note to change your designated port and and username. Now you should be inside the container and see your work space set up and run the command:

```bash
jupyter-lab --allow-root
```
You can connect to the notebook server at 130.162.193.21:<your_designated_port>

Don't forget to login to wandb and you're good to go! 

Some tips:

- Run nvidia-smi to see which GPU's are in use. 
- pass environment argument before starting a process `CUDA_VISIBLE_DEVICES=0` to use only GPU=0.


## Optional - Volume Access

Mount the volume as follows:
```bash
sudo iscsiadm -m node -o new -T iqn.2015-12.com.oracleiaas:61329e5f-3c77-44e8-a9b0-bbb5531217f4 -p 169.254.2.2:3260
sudo iscsiadm -m node -o update -T iqn.2015-12.com.oracleiaas:61329e5f-3c77-44e8-a9b0-bbb5531217f4 -n 
```

The volume is accesible viahome/ubuntu/work.
