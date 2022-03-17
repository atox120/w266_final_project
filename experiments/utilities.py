# A file of helper functions

def CopyCheckpointFolder(resume_from_checkpoint,output_directory):
    '''
    copy the files neede in the output directory with a name checkpoint-resume_from_checkpoint
    the folder will be delete as we only save limited checkpoint, but its exactly what we wanna
    return the new resume_from_checkpoint folder address
    '''
    import shutil
    import os    
    #set the folder adress
    new_resume_from_checkpoint=output_directory+'/checkpoint-resume_from_checkpoint'
    
    #delete the existing folder
    print('deleting the existing checkpoint-resume_from_checkpoint folder')
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)
    shutil.rmtree(new_resume_from_checkpoint,ignore_errors=True)
    
    #copy the folder
    print('copying from {resume_from_checkpoint} into checkpoint-resume_from_checkpoint folder')
    os.makedirs(os.path.dirname(new_resume_from_checkpoint), exist_ok=True)
    shutil.copytree(resume_from_checkpoint,new_resume_from_checkpoint, symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False,  dirs_exist_ok=False)
        
    #remove the optimizer, scheduler and trainer_state

    for i in ['optimizer.pt','scheduler.pt','trainer_state.json']:
        os.remove(f"{new_resume_from_checkpoint}/{i}")

    
    return new_resume_from_checkpoint