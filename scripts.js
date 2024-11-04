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
        'https://media.licdn.com/dms/image/v2/D5603AQGA7P530nogyw/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1695441541808?e=1736380800&v=beta&t=pXuxrcKDpeljeU-3CjSvyrwbfGIvA9MpWemKWFSXc84',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Michael_Jordan_in_2014.jpg/220px-Michael_Jordan_in_2014.jpg',
        // Add more URLs as needed
    ];

     // Fetch and process reference images
     const labeledDescriptors = await Promise.all(
        refImages.map(async (url, index) => {
            const img = await faceapi.fetchImage(url);
            const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
            if (!detections) return null;
            return new faceapi.LabeledFaceDescriptors(`Person ${index + 1}`, [detections.descriptor]);
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