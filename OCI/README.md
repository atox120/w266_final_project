# w266_final_project

## Cloud Access

Access via the ssh key is in a terminal. Note this assumes that the .key file is in the working directory. 
'''bash
ssh -i ssh-key-2022-02-02.key ubuntu@132.145.146.4
'''
Don't forget to have the permissions on the key set via:

```bash
chmod 400 ssh-key-2022-02-02.key
```
## Volume Access

Mount the volume as follows:
```bash
sudo iscsiadm -m node -o new -T iqn.2015-12.com.oracleiaas:61329e5f-3c77-44e8-a9b0-bbb5531217f4 -p 169.254.2.2:3260
sudo iscsiadm -m node -o update -T iqn.2015-12.com.oracleiaas:61329e5f-3c77-44e8-a9b0-bbb5531217f4 -n 
```
