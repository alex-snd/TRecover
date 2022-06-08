---
title: "👀 Demo"

hide:

- navigation
- toc

---

#      

!!! important "<font size="+1">👀 Play with demo</font>"

    === "🤗 Hugging Face"
        <font size="+1">You can play with a pre-trained model hosted [here](https://huggingface.co/spaces/alex-snd/TRecover).</font>

    === "🐳 Docker Compose"
        <font size="+1">Use the command below to run the service via Docker Compose.</font>

        === "Pull from Docker Hub"

            === "standalone"
                ```shell
                docker-compose -f docker/compose/all-in-one-service.yml up
                ```

            === "scallable"
                ```shell
                docker-compose -f docker/compose/scalable-service.yml up
                ```
        
        === "Build from source"
            
            === "standalone"
                ```shell
                docker-compose -f docker/compose/all-in-one-service-duild.yml up --build
                ```

            === "scallable"
                ```shell
                docker-compose -f docker/compose/scalable-service-build.yml up --build
                ```
        <br>
        <font size="+1">:soon: TODO Play with docker</font>

    === "💻 Local"
        <font size="+1">
            To run the service locally, docker must be installed.<br><br>
            First you need to download pretrained model:
        </font>

        ```shell
        trecover download artifacts
        ```
        
        <font size="+1">Then you can run the service:</font>
        
        ```shell
        trecover up
        ```
        
        <font size="+1">
            :grey_question: For more information use ``trecover --help`` or read the 
            <a href="https://alex-snd.github.io/TRecover/src/trecover/app/cli/">reference</a>.
        </font>
        
        
    === ":fontawesome-solid-fire: Torch Hub"
        <font size="+1">You can load the pre-trained model directly from the python script.</font>
        
        ```python
        import pytorch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.hub.load('alex-snd/TRecover', model='trecover', device=device, version='latest')
        ```

        <font size="+1">To print a help message and available pre-trained models run this:</font>
        
        ```python
        import pytorch
        
        print(torch.hub.help('alex-snd/TRecover', model='trecover', force_reload=True))
        ```


??? question "Do you want to challenge yourself ? 😉"
    :soon: TODO

