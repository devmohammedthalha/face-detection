console.log(faceapi)

const run = async()=>{
    //loading the models is going to use await
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    })
    const videoFeedEl = document.getElementById('video-feed')
    videoFeedEl.srcObject = stream
    //we need to load our models
    // pre-trained machine learning for our facial detection!
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'), 
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models'),
    ])

    //make the canvas the same size and in the same location
    // as our video feed
    const canvas = document.getElementById('canvas')
    canvas.style.left = videoFeedEl.offsetLeft
    canvas.style.top = videoFeedEl.offsetTop
    canvas.height = videoFeedEl.height 
    canvas.width = videoFeedEl.width

    // Load multiple reference images
    const refImages = [
        'https://media.licdn.com/dms/image/v2/D5603AQGA7P530nogyw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1695441541808?e=1742428800&v=beta&t=m0e6abzGp3Mzt5BJ3cwVszM4VZ-nFyr41oLPIiQvQ64',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Michael_Jordan_in_2014.jpg/220px-Michael_Jordan_in_2014.jpg',
        // "https://scontent.fcjb1-2.fna.fbcdn.net/v/t39.30808-6/464194022_8468681629896996_187184252050306204_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=waE4ZHu8LBIQ7kNvgHtNN4p&_nc_zt=23&_nc_ht=scontent.fcjb1-2.fna&_nc_gid=AYlU1TlVuogc3dnK1FAFLIk&oh=00_AYBsfxo2WrVcNGDsmC9JhWDy8VKa2kZ4hH6Gqu9uOepzlA&oe=673024F1",
        // "https://media.licdn.com/dms/image/v2/D5603AQHtqbXuNpjizQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1730732892409?e=1736380800&v=beta&t=t_1BKrkPHbdGd0G3Aq-HMXLw__SHHNAI39NGpOIPlVI"
        // Add more URLs as needed
    ];

    const refNames = ["Thalha", "Jordan"]
     // Fetch and process reference images
     const labeledDescriptors = await Promise.all(
        refImages.map(async (url, index) => {
            const img = await faceapi.fetchImage(url);
            const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
            if (!detections) return null;
            return new faceapi.LabeledFaceDescriptors(`${refNames[index]}`, [detections.descriptor]);
        })
    ).then(descriptors => descriptors.filter(desc => desc !== null));

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    // facial detection with points
    setInterval(async()=>{
        // get the video feed and hand it to detectAllFaces method
        let faceAIData = await faceapi.detectAllFaces(videoFeedEl).withFaceLandmarks().withFaceDescriptors().withAgeAndGender().withFaceExpressions()
        // console.log(faceAIData)
        // we have a ton of good facial detection data in faceAIData
        // faceAIData is an array, one element for each face

        // draw on our face/canvas
        //first, clear the canvas
        canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height)
        // draw our bounding box
        faceAIData = faceapi.resizeResults(faceAIData,videoFeedEl)
        faceapi.draw.drawDetections(canvas,faceAIData)
        faceapi.draw.drawFaceLandmarks(canvas,faceAIData)
        faceapi.draw.drawFaceExpressions(canvas,faceAIData)

        // ask AI to guess age and gender with confidence level
        faceAIData.forEach(face=>{
            const { age, gender, genderProbability, detection, descriptor } = face
            const genderText = `${gender} - ${Math.round(genderProbability*100)/100*100}`
            const ageText = `${Math.round(age)} years`
            const textField = new faceapi.draw.DrawTextField([genderText,ageText],face.detection.box.topRight)
            textField.draw(canvas)

            // let label = faceMatcher.findBestMatch(descriptor).toString()
            // console.log(label)
            // let options = {label: "Thalha"}
            // if(label.includes("unknown")){
            //     options = {label: "Unknown subject..."}
            // }
            // const drawBox = new faceapi.draw.DrawBox(detection.box, options)
            // drawBox.draw(canvas)

            const label = faceMatcher.findBestMatch(descriptor).toString();
            const drawBox = new faceapi.draw.DrawBox(detection.box, { label: label.includes("unknown") ? "Unknown subject..." : label });
            drawBox.draw(canvas); 
        })
        

    },200)

}

run()