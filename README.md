
![LogoSPM](https://github.com/portrai-io/spatialspm/assets/103564171/b036ac99-2ed9-4369-8fa8-0185dc9d2f5b)

# SpatialSPM

The **SpatialSPM** is a  tool designed for the analysis and comparison of gene expression patterns in spatial transcriptomic (ST) data. It adeptly handles the complexity of comparing spatial gene expression across different samples by leveraging image format reconstruction and spatial registration techniques.
By reconstructing the data into image format and incorporating spatial registration, it allows for direct comparison of gene expression across different samples. The app produces statistical parametric maps to identify significantly different regions between samples, providing insights into spatial expression patterns in various biological conditions.

## Key Features
- **Data Reconstruction**: Converts spatial transcriptomic data into an image matrix for using image analytic pipelines such as registrations.
- **Spatial Registration**: Facilitates the direct comparison of gene expression across various samples.
- **Statistical Parametric Maps**: Generates maps to identify statistical significant differences in features as a pixelwise manner, offering deep insights into spatial expression patterns under different biological conditions.

## Getting Started

### How to Use
1. Clone the repository: `git clone https://github.com/example/spatialspm-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download ST data to './Buzzi_HemeHpx' from GSE182127 as a sample data
4. Run `Imagizer_hvg_example.ipynb` notebook


## Acknowledgments

## Reference
Please cite "Ohn J, Seo MK, Park J, Lee D, Choi H. SpatialSPM: Statistical parametric mapping for the comparison of gene expression pattern images in multiple spatial transcriptomic datasets. bioRxiv. 2023:2023-06."

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
