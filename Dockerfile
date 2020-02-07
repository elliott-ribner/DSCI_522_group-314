# build this distribution locally by running docker build --tag <tagname> <path to this file>
# install rocker tidy verse image - includes r and tidyverse
FROM rocker/tidyverse
#update apt-get
RUN apt-get update

# install r dependencies
RUN apt-get update -qq && apt-get -y --no-install-recommends install \
  && install2.r --error \
    --deps TRUE \
    tidyverse \
    testthat \
    docopt

# install the anaconda distribution of python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda update -n base -c defaults conda

# set the path so conda and anaconda can be used
ENV PATH /opt/conda/bin:$PATH

# install python dependencies
RUN conda install -c anaconda -y docopt
RUN apt-get install -y python3-pandas
RUN apt-get install -y python3-sklearn python3-sklearn-lib
RUN conda install -c conda-forge altair vega_datasets
RUN conda install -c anaconda -y numpy
RUN pip install -U imbalanced-learn
RUN conda install -c anaconda -y seaborn
RUN conda install -c conda-forge -y matplotlib

# default to using entering docker terminal rather than GUI
CMD ["/bin/bash"]
