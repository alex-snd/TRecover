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
                trecover download artifacts
                docker-compose -f docker/compose/all-in-one-service-duild.yml up --build
                ```

            === "scallable"
                ```shell
                trecover download artifacts
                docker-compose -f docker/compose/scalable-service-build.yml up --build
                ```
        <br>
        <font size="+1">
            You can also try the <a href="https://labs.play-with-docker.com/"> Play with Docker</a> service mentioned in
            the official 
            <a href="https://docs.docker.com/get-started/04_sharing_app/#run-the-image-on-a-new-instance">
                docker documentation
            </a>. 
        </font>

    === "💻 Local"
        <font size="+1">
            To run the service locally, 
            <a href="https://alex-snd.github.io/TRecover/getting_started/installation/">trecover</a> and 
            <a href="https://docs.docker.com/engine/install/">docker</a> must be installed.<br><br>
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
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.hub.load('alex-snd/TRecover', model='trecover', device=device, version='latest')
        ```

        <font size="+1">To print a help message and available pre-trained models run this:</font>
        
        ```python
        import torch
        
        print(torch.hub.help('alex-snd/TRecover', model='trecover', force_reload=True))
        ```


??? question "Do you want to challenge yourself ? 😉"
    
    <font size="+1">Try to read meaningful text in these columns:</font>
    <div align="center">
        <table align="center">
           <tr align="center"><td>a</td><td>d</td><td>f</td><td>o</td><td>n</td><td>p</td><td>p</td><td>u</td><td>w</td><td>f</td><td>o</td><td>u</td><td>d</td><td>d</td><td>y</td><td>k</td><td>d</td><td>d</td><td>a</td><td>u</td><td>n</td><td>t</td><td>r</td><td>y</td><td>x</td><td>g</td><td>n</td><td>k</td><td>w</td><td>n</td><td>t</td><td>n</td><td>t</td><td>t</td><td>a</td><td>t</td><td>t</td><td>u</td><td>r</td><td>e</td><td>t</td><td>g</td><td>t</td><td>x</td><td>r</td><td>u</td><td>e</td><td>w</td><td>r</td><td>t</td><td>h</td><td>n</td><td>x</td><td>o</td><td>d</td><td>v</td><td>t</td><td>i</td><td>t</td><td>i</td><td>o</td><td>t</td><td>p</td><td>f</td><td>m</td><td>o</td><td>b</td><td>k</td><td>j</td><td>z</td><td>t</td><td>g</td><td>d</td><td>s</td><td>c</td><td>y</td><td>w</td><td>w</td><td>w</td><td>t</td><td>d</td><td>x</td><td>h</td><td>k</td><td>n</td><td>d</td><td>p</td><td>d</td><td>a</td><td>r</td><td>d</td><td>x</td><td>d</td><td>n</td><td>g</td><td>t</td><td>h</td><td>p</td><td>u</td><td>r</td><td>w</td><td>u</td><td>n</td><td>d</td><td>d</td><td>n</td><td>z</td></tr>
           <tr align="center"><td> </td><td>s</td><td>p</td><td>f</td><td>g</td><td> </td><td>g</td><td>e</td><td>a</td><td>r</td><td> </td><td> </td><td>n</td><td>g</td><td>r</td><td>h</td><td>e</td><td>h</td><td>o</td><td> </td><td>w</td><td>l</td><td> </td><td>z</td><td>w</td><td>c</td><td>g</td><td>l</td><td>i</td><td>f</td><td> </td><td>o</td><td>p</td><td>c</td><td>e</td><td>w</td><td>w</td><td>r</td><td>e</td><td> </td><td> </td><td>y</td><td> </td><td>g</td><td>c</td><td>b</td><td> </td><td>l</td><td> </td><td>y</td><td> </td><td>w</td><td>d</td><td> </td><td>n</td><td> </td><td>h</td><td>k</td><td>k</td><td> </td><td> </td><td>w</td><td>s</td><td>r</td><td>o</td><td>e</td><td> </td><td>u</td><td>s</td><td>i</td><td>n</td><td>e</td><td>g</td><td> </td><td>i</td><td>s</td><td>h</td><td>n</td><td>p</td><td>h</td><td>n</td><td>v</td><td>u</td><td>v</td><td>b</td><td>o</td><td>b</td><td>u</td><td> </td><td> </td><td>z</td><td>a</td><td>u</td><td>x</td><td>p</td><td>p</td><td> </td><td>i</td><td>i</td><td> </td><td>b</td><td>i</td><td> </td><td>w</td><td>k</td><td>k</td><td>s</td></tr>
           <tr align="center"><td> </td><td> </td><td>z</td><td>e</td><td>k</td><td> </td><td>h</td><td> </td><td>q</td><td>l</td><td> </td><td> </td><td>x</td><td>r</td><td>t</td><td>o</td><td>u</td><td>y</td><td> </td><td> </td><td>z</td><td>e</td><td> </td><td>p</td><td>a</td><td>e</td><td>e</td><td>q</td><td>q</td><td>p</td><td> </td><td> </td><td>g</td><td>f</td><td> </td><td>a</td><td> </td><td> </td><td> </td><td> </td><td> </td><td>u</td><td> </td><td>o</td><td> </td><td>h</td><td> </td><td>e</td><td> </td><td> </td><td> </td><td>p</td><td>s</td><td> </td><td>h</td><td> </td><td>i</td><td>c</td><td>u</td><td> </td><td> </td><td>n</td><td> </td><td>i</td><td> </td><td> </td><td> </td><td>i</td><td>q</td><td> </td><td>y</td><td> </td><td>r</td><td> </td><td>o</td><td> </td><td>i</td><td>e</td><td> </td><td>l</td><td>p</td><td>p</td><td>r</td><td>e</td><td>f</td><td>m</td><td>e</td><td>s</td><td> </td><td> </td><td>r</td><td>c</td><td> </td><td>k</td><td>u</td><td>i</td><td> </td><td>e</td><td>m</td><td> </td><td>h</td><td> </td><td> </td><td>g</td><td>e</td><td>w</td><td> </td></tr>
           <tr align="center"><td> </td><td> </td><td>i</td><td> </td><td>h</td><td> </td><td>l</td><td> </td><td> </td><td>q</td><td> </td><td> </td><td>r</td><td> </td><td>s</td><td> </td><td>a</td><td>s</td><td> </td><td> </td><td>h</td><td> </td><td> </td><td>e</td><td>b</td><td> </td><td>r</td><td>t</td><td> </td><td>r</td><td> </td><td> </td><td>q</td><td>h</td><td> </td><td>s</td><td> </td><td> </td><td> </td><td> </td><td> </td><td>t</td><td> </td><td> </td><td> </td><td>q</td><td> </td><td> </td><td> </td><td> </td><td> </td><td>e</td><td>c</td><td> </td><td>e</td><td> </td><td>r</td><td>q</td><td>o</td><td> </td><td> </td><td> </td><td> </td><td>o</td><td> </td><td> </td><td> </td><td>q</td><td> </td><td> </td><td>e</td><td> </td><td>q</td><td> </td><td>e</td><td> </td><td>c</td><td> </td><td> </td><td>o</td><td>r</td><td>y</td><td>l</td><td>a</td><td>p</td><td>e</td><td>a</td><td>m</td><td> </td><td> </td><td>q</td><td>e</td><td> </td><td>u</td><td>l</td><td> </td><td> </td><td> </td><td> </td><td> </td><td>r</td><td> </td><td> </td><td>p</td><td>i</td><td>h</td><td> </td></tr>
           <tr align="center"><td> </td><td> </td><td>q</td><td> </td><td>b</td><td> </td><td> </td><td> </td><td> </td><td>j</td><td> </td><td> </td><td>m</td><td> </td><td>c</td><td> </td><td>s</td><td>c</td><td> </td><td> </td><td>y</td><td> </td><td> </td><td> </td><td>c</td><td> </td><td> </td><td>o</td><td> </td><td>s</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>s</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>a</td><td> </td><td> </td><td>l</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>t</td><td> </td><td> </td><td> </td><td> </td><td>s</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>s</td><td>r</td><td>m</td><td>j</td><td> </td><td> </td><td>j</td><td>o</td><td> </td><td> </td><td>s</td><td> </td><td> </td><td>a</td><td>c</td><td> </td><td> </td><td> </td><td> </td><td> </td><td>j</td><td> </td><td> </td><td> </td><td>o</td><td>q</td><td> </td></tr>
           <tr align="center"><td> </td><td> </td><td> </td><td> </td><td>o</td><td> </td><td> </td><td> </td><td> </td><td>o</td><td> </td><td> </td><td> </td><td> </td><td>o</td><td> </td><td>m</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>o</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>m</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>q</td><td> </td><td> </td><td>m</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>b</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>c</td><td>t</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>m</td><td> </td><td> </td><td>i</td><td>m</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>s</td><td> </td></tr>
        </table>
    </div>
    <br>
    
    <font size="+1">And see what the pre-trained model will recover:</font><br>
    
    <font size="+1">1. Copy these columns</font>
    ```
    a ds fpziq ofe ngkhbo p pghl ue waq frlqjo o u dnxrm dgr yrtsco kho deuasm dhysc ao u nwzhy tle r yzpe xwabc gce nger klqto wiq nfprso t no tpgq tcfh ae twas tw ur re e t gyutsm t xgo rc ubhq e wle r ty h nwpeaq xdsc o dnhelm v thir ikcq tkuo i o twn ps frio mo oe b kuiqtb jsq zi tnye ge dgrqs s cioe ys whic wne wp thlo dnprsc xvpyrt hurlm kveaj nbfp dome pbeaj dusmo a r dzrqsm xace du nxkuai gpulcm tpi h pie uim r wbhrj ui n dwgp dkeio nkwhqs zs
    ```
    
    <font size="+1">2. Open the dashboard hosted <a href="https://huggingface.co/spaces/alex-snd/TRecover">here</a></font>
    <br>    
    <font size="+1">3. In the sidebar, select "Noisy columns" as an input type</font>
    <br>    
    <font size="+1">4. Paste the copied columns into the input field</font>
    <br>    
    <font size="+1">5. Click the "Recover" button 🎉</font>


