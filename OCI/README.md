# w266_final_project

## Cloud Access

Access via the ssh key is in a terminal. Note this assumes that the .key file is in the working directory. 

```bash
ssh -i ssh-key-GPU.key ubuntu@168.138.21.187
```

Don't forget to have the read permissions on the key set via:

```bash
chmod 400 ssh-key-GPU.key
```

## Usage

Once inside the `src/source_chkpts` folder contains all model checkpoints. 

### Running a Jupyter Notebook:

Port designation:

- Wanyu: 8886
- Rathin: 8882
- Alex: 8888

The container is prebuilt using the dockerfile in src/Dockerfile. The image name is `atox120/w266_fp_torch:v2`. To run the container please use the following command:



## Optional - Volume Access

Mount the volume as follows:
```bash
sudo iscsiadm -m node -o new -T iqn.2015-12.com.oracleiaas:61329e5f-3c77-44e8-a9b0-bbb5531217f4 -p 169.254.2.2:3260
sudo iscsiadm -m node -o update -T iqn.2015-12.com.oracleiaas:61329e5f-3c77-44e8-a9b0-bbb5531217f4 -n 
```

The volume is accesible viahome/ubuntu/work.
