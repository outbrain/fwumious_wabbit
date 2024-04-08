#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
#![allow(redundant_semicolons)]
#![allow(dead_code, unused_imports)]

use flate2::read::MultiGzDecoder;
use std::collections::VecDeque;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use zstd::stream::read::Decoder as ZstdDecoder;

extern crate blas;
extern crate half;
extern crate intel_mkl_src;

#[macro_use]
extern crate nom;
extern crate core;

use fw::cache::RecordCache;
use fw::feature_buffer::FeatureBufferTranslator;
use fw::hogwild::HogwildTrainer;
use fw::model_instance::{ModelInstance, Optimizer};
use fw::multithread_helpers::BoxedRegressorTrait;
use fw::parser::VowpalParser;
use fw::persistence::{
    new_regressor_from_filename, save_regressor_to_filename, save_sharable_regressor_to_filename,
};
use fw::regressor::{get_regressor_with_weights, Regressor};
use fw::serving::Serving;
use fw::vwmap::VwNamespaceMap;
use fw::{cmdline, feature_buffer, logging_layer, regressor};

fn main() {
    logging_layer::initialize_logging_layer();

    if let Err(e) = main_fw_loop() {
        log::error!("Global error: {:?}", e);
        std::process::exit(1)
    }
}

fn create_buffered_input(input_filename: &str) -> Box<dyn BufRead> {
    // Handler for different (or no) compression types

    let input = File::open(input_filename).expect("Could not open the input file.");

    let input_format = Path::new(&input_filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .expect("Failed to get the file extension.");

    match input_format {
        "gz" => {
            let gz_decoder = MultiGzDecoder::new(input);
            let reader = io::BufReader::new(gz_decoder);
            Box::new(reader)
        }
        "zst" => {
            let zstd_decoder = ZstdDecoder::new(input).unwrap();
            let reader = io::BufReader::new(zstd_decoder);
            Box::new(reader)
        }
        "vw" => {
            let reader = io::BufReader::new(input);
            Box::new(reader)
        }
        _ => {
            panic!("Please specify a valid input format (.vw, .zst, .gz)");
        }
    }
}

fn build_cache_without_training(cl: clap::ArgMatches) -> Result<(), Box<dyn Error>> {
    /*! A method that enables creating the cache file without training the first model instance.
    This is done in order to reduce building time of the cache and running the first model instance multi threaded. */
    // We'll parse once the command line into cl and then different objects will examine it
    let input_filename = cl.value_of("data").expect("--data expected");
    let vw_namespace_map_filepath = Path::new(input_filename)
        .parent()
        .expect("Couldn't access path given by --data")
        .join("vw_namespace_map.csv");

    let vw: VwNamespaceMap = VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
    let mut cache = RecordCache::new(input_filename, true, &vw);
    let input = File::open(input_filename)?;

    let mut bufferred_input = create_buffered_input(input_filename);
    let mut pa = VowpalParser::new(&vw);
    let mut example_num = 0;
    loop {
        let reading_result;
        let buffer: &[u32];
        if !cache.reading {
            reading_result = pa.next_vowpal(&mut bufferred_input);
            buffer = match reading_result {
                Ok([]) => break, // EOF
                Ok(buffer2) => buffer2,
                Err(_e) => return Err(_e),
            };
            if cache.writing {
                cache.push_record(buffer)?;
            }
        } else {
            reading_result = cache.get_next_record();
            match reading_result {
                Ok([]) => break, // EOF
                Ok(buffer) => buffer,
                Err(_e) => return Err(_e),
            };
        }
        example_num += 1;
    }

    log::info!("Built cache only, exiting.");
    cache.write_finish()?;
    Ok(())
}

fn main_fw_loop() -> Result<(), Box<dyn Error>> {
    // We'll parse once the command line into cl and then different objects will examine it
    let cl = cmdline::parse();
    if cl.is_present("build_cache_without_training") {
        return build_cache_without_training(cl);
    }
    // Where will we be putting perdictions (if at all)
    let mut predictions_file = match cl.value_of("predictions") {
        Some(filename) => Some(BufWriter::new(File::create(filename)?)),
        None => None,
    };

    let testonly = cl.is_present("testonly");
    let quantize_weights = cl.is_present("weight_quantization");
    let final_regressor_filename = cl.value_of("final_regressor");
    let output_pred_sto: bool = cl.is_present("predictions_stdout");
    if let Some(filename) = final_regressor_filename {
        if !cl.is_present("save_resume") {
            return Err("You need to use --save_resume with --final_regressor, for vowpal wabbit compatibility")?;
        }
        log::info!("final_regressor = {}", filename);
    };

    let inference_regressor_filename = cl.value_of("convert_inference_regressor");
    if let Some(filename) = inference_regressor_filename {
        log::info!("inference_regressor = {}", filename);
    };

    /* setting up the pipeline, either from command line or from existing regressor */
    // we want heal-allocated objects here

    if cl.is_present("daemon") {
        let filename = cl
            .value_of("initial_regressor")
            .expect("Daemon mode only supports serving from --initial regressor");
        log::info!("initial_regressor = {}", filename);
        let (mi2, vw2, re_fixed) = new_regressor_from_filename(filename, true, Option::Some(&cl))?;

        let mut se = Serving::new(&cl, &vw2, Box::new(re_fixed), &mi2)?;
        se.serve()?;
    } else if cl.is_present("convert_inference_regressor") {
        let filename = cl
            .value_of("initial_regressor")
            .expect("Convert mode requires --initial regressor");
        let (mut mi2, vw2, re_fixed) =
            new_regressor_from_filename(filename, true, Option::Some(&cl))?;
        mi2.optimizer = Optimizer::SGD;
        if cl.is_present("weight_quantization") {
            mi2.dequantize_weights = Some(true);
        }
        if let Some(filename1) = inference_regressor_filename {
            save_regressor_to_filename(filename1, &mi2, &vw2, re_fixed, quantize_weights).unwrap()
        }
    } else {
        let vw: VwNamespaceMap;
        let mut re: Regressor;
        let mut sharable_regressor: BoxedRegressorTrait;
        let mi: ModelInstance;

        if let Some(filename) = cl.value_of("initial_regressor") {
            log::info!("initial_regressor = {}", filename);
            (mi, vw, re) = new_regressor_from_filename(filename, testonly, Option::Some(&cl))?;
            sharable_regressor = BoxedRegressorTrait::new(Box::new(re));
        } else {
            // We load vw_namespace_map.csv just so we know all the namespaces ahead of time
            // This is one of the major differences from vowpal

            let input_filename = cl.value_of("data").expect("--data expected");
            let vw_namespace_map_filepath = Path::new(input_filename)
                .parent()
                .expect("Couldn't access path given by --data")
                .join("vw_namespace_map.csv");
            vw = VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
            mi = ModelInstance::new_from_cmdline(&cl, &vw)?;
            re = get_regressor_with_weights(&mi);
            sharable_regressor = BoxedRegressorTrait::new(Box::new(re));
        };

        let input_filename = cl.value_of("data").expect("--data expected");
        let mut cache = RecordCache::new(input_filename, cl.is_present("cache"), &vw);
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let mut pb = sharable_regressor.new_portbuffer();

        let predictions_after: u64 = match cl.value_of("predictions_after") {
            Some(examples) => examples.parse()?,
            None => 0,
        };

        let holdout_after_option: Option<u64> =
            cl.value_of("holdout_after").map(|s| s.parse().unwrap());

        let hogwild_training = cl.is_present("hogwild_training");
        let mut hogwild_trainer = if hogwild_training {
            let hogwild_threads = match cl.value_of("hogwild_threads") {
                Some(hogwild_threads) => hogwild_threads
                    .parse()
                    .expect("hogwild_threads should be integer"),
                None => 16,
            };
            HogwildTrainer::new(sharable_regressor.clone(), &mi, hogwild_threads)
        } else {
            HogwildTrainer::default()
        };

        let prediction_model_delay: u64 = match cl.value_of("prediction_model_delay") {
            Some(delay) => delay.parse()?,
            None => 0,
        };

        let mut delayed_learning_fbs: VecDeque<feature_buffer::FeatureBuffer> =
            VecDeque::with_capacity(prediction_model_delay as usize);

        let mut bufferred_input = create_buffered_input(input_filename);
        let mut pa = VowpalParser::new(&vw);

        let now = Instant::now();
        let mut example_num = 0;
        loop {
            let reading_result;
            let buffer: &[u32];
            if !cache.reading {
                reading_result = pa.next_vowpal(&mut bufferred_input);
                buffer = match reading_result {
                    Ok([]) => break, // EOF
                    Ok(buffer2) => buffer2,
                    Err(_e) => return Err(_e),
                };
                if cache.writing {
                    cache.push_record(buffer)?;
                }
            } else {
                reading_result = cache.get_next_record();
                buffer = match reading_result {
                    Ok([]) => break, // EOF
                    Ok(buffer) => buffer,
                    Err(_e) => return Err(_e),
                };
            }
            example_num += 1;
            let mut prediction: f32 = 0.0;

            if prediction_model_delay == 0 {
                let update = match holdout_after_option {
                    Some(holdout_after) => !testonly && example_num < holdout_after,
                    None => !testonly,
                };
                if hogwild_training && update {
                    hogwild_trainer.digest_example(Vec::from(buffer));
                } else {
                    fbt.translate(buffer, example_num);
                    prediction = sharable_regressor.learn(&fbt.feature_buffer, &mut pb, update);
                }
            } else {
                fbt.translate(buffer, example_num);
                if example_num > predictions_after {
                    prediction = sharable_regressor.learn(&fbt.feature_buffer, &mut pb, false);
                }
                delayed_learning_fbs.push_back(fbt.feature_buffer.clone());
                if (prediction_model_delay as usize) < delayed_learning_fbs.len() {
                    let delayed_buffer = delayed_learning_fbs.pop_front().unwrap();
                    sharable_regressor.learn(&delayed_buffer, &mut pb, !testonly);
                }
            }

            if example_num > predictions_after {
                if output_pred_sto {
                    println!("{:.6}", prediction);
                }

                match predictions_file.as_mut() {
                    Some(file) => writeln!(file, "{:.6}", prediction)?,
                    None => {}
                }
            }
        }
        cache.write_finish()?;

        if hogwild_training {
            hogwild_trainer.block_until_workers_finished();
        }
        let elapsed = now.elapsed();
        log::info!("Elapsed: {:.2?} rows: {}", elapsed, example_num);

        if let Some(filename) = final_regressor_filename {
            save_sharable_regressor_to_filename(
                filename,
                &mi,
                &vw,
                sharable_regressor,
                quantize_weights,
            )
            .unwrap()
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs::File;
    use std::io::{self, BufReader, Read, Write};
    use tempfile::Builder as TempFileBuilder;
    use tempfile::NamedTempFile;
    use zstd::stream::Encoder as ZstdEncoder;

    fn create_temp_file_with_contents(
        extension: &str,
        contents: &[u8],
    ) -> io::Result<NamedTempFile> {
        let temp_file = TempFileBuilder::new()
            .suffix(&format!(".{}", extension))
            .tempfile()?;
        temp_file.as_file().write_all(contents)?;
        Ok(temp_file)
    }

    fn create_gzipped_temp_file(contents: &[u8]) -> io::Result<NamedTempFile> {
        let temp_file = TempFileBuilder::new().suffix(".gz").tempfile()?;
        let gz = GzEncoder::new(Vec::new(), Compression::default());
        let mut gz_writer = io::BufWriter::new(gz);
        gz_writer.write_all(contents)?;
        let gz = gz_writer.into_inner()?.finish()?;
        temp_file.as_file().write_all(&gz)?;
        Ok(temp_file)
    }

    fn create_zstd_temp_file(contents: &[u8]) -> io::Result<NamedTempFile> {
        let temp_file = TempFileBuilder::new().suffix(".zst").tempfile()?;
        let mut zstd_encoder = ZstdEncoder::new(Vec::new(), 1)?;
        zstd_encoder.write_all(contents)?;
        let encoded_data = zstd_encoder.finish()?;
        temp_file.as_file().write_all(&encoded_data)?;
        Ok(temp_file)
    }

    // Test for uncompressed file ("vw" extension)
    #[test]
    fn test_uncompressed_file() {
        let contents = b"Sample text for uncompressed file.";
        let temp_file =
            create_temp_file_with_contents("vw", contents).expect("Failed to create temp file");
        let mut reader = create_buffered_input(temp_file.path().to_str().unwrap());

        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read from the reader");
        assert_eq!(
            buffer, contents,
            "Contents did not match for uncompressed file."
        );
    }

    // Test for gzipped files ("gz" extension)
    #[test]
    fn test_gz_compressed_file() {
        let contents = b"Sample text for gzipped file.";
        let temp_file =
            create_gzipped_temp_file(contents).expect("Failed to create gzipped temp file");
        let mut reader = create_buffered_input(temp_file.path().to_str().unwrap());

        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read from the reader");
        assert_eq!(buffer, contents, "Contents did not match for gzipped file.");
    }

    // Test for zstd compressed files ("zst" extension)
    #[test]
    fn test_zstd_compressed_file() {
        let contents = b"Sample text for zstd compressed file.";
        let temp_file = create_zstd_temp_file(contents).expect("Failed to create zstd temp file");
        let mut reader = create_buffered_input(temp_file.path().to_str().unwrap());

        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read from the reader");
        assert_eq!(
            buffer, contents,
            "Contents did not match for zstd compressed file."
        );
    }

    // Test for unsupported file format
    #[test]
    #[should_panic(expected = "Please specify a valid input format (.vw, .zst, .gz)")]
    fn test_unsupported_file_format() {
        let contents = b"Some content";
        let temp_file =
            create_temp_file_with_contents("txt", contents).expect("Failed to create temp file");
        let _reader = create_buffered_input(temp_file.path().to_str().unwrap());
    }
}
