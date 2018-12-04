function intervalFunc() {
    var mongodb = require('mongodb');
        var MongoClient = mongodb.MongoClient;
        var url = 'mongodb://giuxe:123456789bA@ds133113.mlab.com:33113/giuxe';
        MongoClient.connect(url,{useNewUrlParser: true} , function (err, db) {
          if (err) {
            console.log('Unable to connect to the mongoDB server. Error:', err);
          } 
    var idx = 0;
    var fs = require('fs'); 



    fs.readdir("StoreImages", function(err, files) {
        if (err) {
            console.log("faillll");
            throw err;
        } 
        else {
            var PythonShell = require('python-shell');

            var options = {
            scriptPath: 'E:/IoTProject/Machine-Learning-Plate-Recognition',
            mode: 'text',
            pythonPath: 'C:/Users/GIAPHUC/Anaconda3/envs/keras/python.exe', 
            pythonOptions: ['-u'],
            // make sure you use an absolute path for scriptPath
            //args: ['value1', 'value2', 'value3']
            };
            PythonShell.PythonShell.run('PredictCNN.py', options, function (err, results) {
                if (err) throw err;
                // results is an array consisting of messages collected during execution
                console.log('results: %j', results);
              });
        }
    });





    fs.readdir("StoreTxt", function(err, files) {
        if (err) {
            console.log("faillll");
            throw err;
        } 
        else {
            var dbo = db.db("giuxe");
            //console.log(dbo.collection("xe").find({}));
            if(files.length != 0)
                dbo.collection("xe").deleteMany();
            //var keepFilesLength = files.length;
            files.forEach(file => {
                var path = "StoreTxt\\"+file;
                fs.readFile(path, 'utf8', function(err, data) {
                    var dbo = db.db("giuxe");
                    var myobj = { ID: data };
                    dbo.collection("xe").insertOne(myobj, function(err, res) {
                        if (err) throw err;
                        console.log(file+" document inserted");
                        
                      });
                     
                    
                });
                fs.unlinkSync(path);
              });
            // while (keepFilesLength != 0 ){
                
                
            // }
        }
    });

    });
}

setInterval(intervalFunc, 5000);