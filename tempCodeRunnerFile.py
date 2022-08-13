if len(os.listdir(path)) == 0:
            
                file.save(os.path.join(app.config['TRAIN_FOLDER'], filename))
                for files in os.listdir(path):
                    print(files)
                    os.rename(path + '/' + files,path +'/'+ new_name)
                

            else:
                for files in os.listdir(path):
                    os.remove(path+'/'+files)