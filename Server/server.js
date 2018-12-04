var http = require('http');
var formidable = require('formidable');
var fs = require('fs');

http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
      var oldpath = files.filetoupload.path;
      var newpath = 'E:/testIOT/Server/StoreImages/' + files.filetoupload.name;
    //   fs.rename(oldpath, newpath, function (err) {
    //     if (err) throw err;
    //     res.write('File uploaded and moved!');
    //     res.end();
    //   });
    fs.readFile(oldpath, function (err, data) {
        if (err) throw err;
        console.log('File read!');

        // Write the file
        fs.writeFile(newpath, data, function (err) {
            if (err) throw err;
            res.write('File uploaded and moved!');
            res.end();
            console.log('File written!');
        });

        // Delete the file
        fs.unlink(oldpath, function (err) {
            if (err) throw err;
            console.log('File deleted!');
        });
    });
 });
  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
    return res.end();
  }
}).listen(8080);

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
                    var str = file.split(".jpg.txt")[0]+"-"+data;
                    var myobj = { ID: str };
                    console.log(str);
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