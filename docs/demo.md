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

        <img src="https://github.com/alex-snd/TRecover/blob/assets/dashboard_demo.gif?raw=true"/>

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
        <font size="+1">
            ```
            a	d	f	o	n	p	p	u	w	f	o	u	d	d	y	k	d	d	a	u	n	t	r	y	x	g	n	k	w	n	t	n	t	t	a	t	t	u	r	e	t	g	t	x	r	u	e	w	r	t	h	n	x	o	d	v	t	i	t	i	o	t	p	f	m	o	b	k	j	z	t	g	d	s	c	y	w	w	w	t	d	x	h	k	n	d	p	d	a	r	d	x	d	n	g	t	h	p	u	r	w	u	n	d	d	n	z
                s	p	f	g		g	e	a	r			n	g	r	h	e	h	o		w	l		z	w	c	g	l	i	f		o	p	c	e	w	w	r	e			y		g	c	b		l		y		w	d		n		h	k	k			w	s	r	o	e		u	s	i	n	e	g		i	s	h	n	p	h	n	v	u	v	b	o	b	u			z	a	u	x	p	p		i	i		b	i		w	k	k	s
                    z	e	k		h		q	l			x	r	t	o	u	y			z	e		p	a	e	e	q	q	p			g	f		a						u		o		h		e				p	s		h		i	c	u			n		i				i	q		y		r		o		i	e		l	p	p	r	e	f	m	e	s			r	c		k	u	i		e	m		h			g	e	w	
                    i		h		l			q			r		s		a	s			h			e	b		r	t		r			q	h		s						t				q						e	c		e		r	q	o					o				q			e		q		e		c			o	r	y	l	a	p	e	a	m			q	e		u	l						r			p	i	h	
                    q		b					j			m		c		s	c			y				c			o		s												s										a			l													t					s								s	r	m	j			j	o			s			a	c						j				o	q	
                            o					o					o		m													o												m										q			m													b													c	t									m			i	m											s
            ```
        </font>
    </div>
    
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


