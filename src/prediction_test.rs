mod block_ffm;
mod block_helpers;
mod block_loss_functions;
mod block_lr;
mod block_misc;
mod block_neural;
mod block_normalize;
mod block_relu;
mod cache;
mod cmdline;
mod consts;
mod feature_buffer;
mod feature_transform_executor;
mod feature_transform_implementations;
mod feature_transform_parser;
mod graph;
mod logging_layer;
mod model_instance;
mod multithread_helpers;
mod optimizer;
mod parser;
mod persistence;
mod port_buffer;
mod regressor;
mod serving;
mod version;
mod vwmap;

extern crate blas;
extern crate intel_mkl_src;

use crate::feature_buffer::FeatureBufferTranslator;
use crate::multithread_helpers::BoxedRegressorTrait;
use crate::parser::VowpalParser;
use crate::port_buffer::PortBuffer;
use shellwords;
use std::ffi::CStr;
use std::io::Cursor;
use std::os::raw::c_char;
use crate::regressor::BlockCache;

fn main() {
    unsafe {
        let command = "outbrain/automl_bin/fw-simd-0.1 -l 0.014 -d /tmp/model_lib/data.vw.gz -p /tmp/model_lib/prediction_log --transform \"comb_country_from_to_lang=Combine(country_v2,from_language_lower,to_language_lower)()\" --transform \"comb_day_time=Combine(day_of_week,user_local_hour)()\" --transform \"comb_feedidx_widget=Combine(feed_idx,widget_id_gen)()\" --transform \"comb_img_deltas_plat=Combine(fcap_img_exp_intervals,platform_upper)()\" --transform \"comb_requestedlstnum_idx_card_idx_widget_id_gen=Combine(requested_lst_num,idx,card_idx,widget_id_gen)()\" --transform \"comb_widget_data_widgets_to_sources=Combine(widget_data_widgets,widget_data_sources)()\" --transform \"comb_window_screen_h_plat=Combine(screen_h,window_h,platform_upper)()\" --transform \"combined_widget_fromsrc_platformupper=Combine(widget_id_gen,from_src,platform_upper)()\" --transform \"fcap_doc_count_log=BinnerLog(fcap_doc_count)(1,5.0)\" --transform \"from_doc_3=Weight(from_doc)(3)\" --transform \"interactions_count_click_sqrt=BinnerSqrtPlain(interactions_count_click)(30,1.0)\" --transform \"interactions_count_page_view_sqrt=BinnerSqrtPlain(interactions_count_page_view)(30,1.0)\" --transform \"logctr1a=BinnerLogRatio(interactions_count_click,interactions_count_page_view)(15,15)\" --transform \"publisher_id_3=Weight(publisher_id)(3)\" --transform \"second_context_ctr_5log=BinnerLog(second_context_ctr)(0.00001,5.0)\" --transform \"view_prob_10Sqrt001=BinnerSqrt(viewability_prob)(0.001,10.0)\" --transform \"viewport_distance_log=BinnerLog(viewport_distance)(10,1)\" --linear ad_title_hash,from_src --linear advertiser_id,from_src --linear advertiser_id,multi_adv_clicked --linear browser_family,from_src,window_w --linear browser_version,publisher_id_3,widget_id_gen --linear browser_version,window_h --linear card_idx,widget_id_gen --linear comb_widget_data_widgets_to_sources --linear from_cat,to_cat --linear from_language_lower,to_language_lower --linear from_src,page_referrer_multiValue --linear from_src,user_interest,widget_id_gen --linear idx,platform_upper,publisher_id_3,widget_id_gen --linear idx,ui_abtest_variant_id --linear image_uuid --linear page_referrer_multiValue --linear page_referrer_multiValue,widget_id_gen --linear requested_lst_num --linear screen_h,window_h --linear second_context_ctr_5log --linear to_doc,user_interest --linear to_src,widget_id_gen --ffm_bit_precision 29 --ffm_field_verbose active_tab,comb_requestedlstnum_idx_card_idx_widget_id_gen --ffm_field_verbose active_tab,second_context_ctr_5log,view_prob_10Sqrt001 --ffm_field_verbose ad_title_hash,advertiser_id,campaign_id,count_clicks_6h,last_pv_hrs --ffm_field_verbose browser_version,os_version,window_w --ffm_field_verbose card_idx,page_referrer_multiValue,publisher_id_3,ui_abtest_id,viewport_distance_log,widget_id_gen --ffm_field_verbose comb_country_from_to_lang --ffm_field_verbose comb_day_time,country_v2 --ffm_field_verbose comb_feedidx_widget --ffm_field_verbose comb_img_deltas_plat --ffm_field_verbose comb_widget_data_widgets_to_sources --ffm_field_verbose comb_window_screen_h_plat --ffm_field_verbose combined_widget_fromsrc_platformupper --ffm_field_verbose conv_clicks,count_pv_24h,interactions_count_click_sqrt,last_click_hrs,multi_cat_clicked --ffm_field_verbose dma_code_v2,postal_code --ffm_field_verbose fcap_doc_count_log --ffm_field_verbose from_cat,from_cat_iab2 --ffm_field_verbose from_doc_3 --ffm_field_verbose from_src --ffm_field_verbose idx,viewport_distance_log --ffm_field_verbose image_uuid,to_doc --ffm_field_verbose interactions_count_page_view_sqrt --ffm_field_verbose logctr1a --ffm_field_verbose multi_cmp_clicked,multi_weighted_adv_converged,user_interest --ffm_field_verbose multi_source_pv --ffm_field_verbose os_family,screen_h,screen_w --ffm_field_verbose page_referrer_multiValue --ffm_field_verbose region_v2 --ffm_field_verbose requested_lst_num,widget_id_gen --ffm_field_verbose to_doc_title_hash,to_doc_top_keyword,top_image_labels --ffm_field_verbose to_src --ffm_field_verbose ui_abtest_variant_id --ffm_k 8 --adaptive --l2 0.0 --hash all --power_t 0.42 --ffm_learning_rate 0.021 --nn_layers 2 --nn 0:activation:relu --nn 0:init:xavier --nn 0:width:75 --nn 1:activation:relu --nn 1:init:xavier --nn 1:width:100 --nn_topology one -i inference_weights.vw --save_resume -f initial_weights.vw --predictions_after 10620078";
        let words = shellwords::split(command).unwrap();
        let cmd_matches = cmdline::create_expected_args().get_matches_from(words);
        let weights_filename = match cmd_matches.value_of("initial_regressor") {
            Some(filename) => filename,
            None => panic!("Cannot resolve input weights file name"),
        };
        let (model_instance, vw_namespace_map, mut regressor) =
            persistence::new_regressor_from_filename(weights_filename, true, Some(&cmd_matches))
                .unwrap();
        let mut feature_buffer_translator = FeatureBufferTranslator::new(&model_instance);
        let mut vw_parser = VowpalParser::new(&vw_namespace_map);
        let mut sharable_regressor = BoxedRegressorTrait::new(Box::new(regressor));
        let mut pb = sharable_regressor.new_portbuffer();

        let full_input_buffer = "|BE BE |Ak Ak0 |Ay Ay |AF AF |AD AD |Af Af |AN AN |BK BK |AV AV |BI BI |AG AG |Ag Ag |Al Al0 |BF BF |Aw Aw |AW AW |Ah Ah |BC BC |BJ BJ |Ai Ai |AS AS |AY AY |Aa Aa |Ae Ae |BG BG |Ar Ar |AH AH |Ax Ax |AM AM0 |Az Az |Ab Ab |Ad Ad |AO AO0 |AC AC |AE AE |AB AB |AZ AZ |At At |AT AT-1 |BL BL |An An |AI AI0 |BD BD |Aj Aj |Au Au |Av Av |Ac Ac |AP AP |AQ AQ0 |AK AK |Ao Ao |AX AXnull |AJ AJ |AU AU-1 |AL AL |AR AR |Am Am |Ap Ap |BH BH |Aq Aq0 |As As0 \n".to_string();
        let mut buffered_input = Cursor::new(full_input_buffer);
        let reading_result = vw_parser.next_vowpal(&mut buffered_input);
        let buffer = match reading_result {
            Ok([]) => panic!("AAAAA"), // EOF
            Ok(buffer2) => buffer2,
            Err(_e) => panic!("BBBB"),
        };
        feature_buffer_translator.translate(buffer, 0);
        let prediction = sharable_regressor
            .predict(&feature_buffer_translator.feature_buffer, &mut pb);
        println!("fw_predict: {}", prediction);

        let context_input_buffer = "|BE BE |Ak Ak0 |Ay Ay |AF AF |AD AD |Af Af |AN AN |BK BK |AV AV |BI BI |AG AG |Ag Ag |Al Al0 |BF BF |Aw Aw |AW AW |Ah Ah |BC BC |BJ BJ |Ai Ai |AS AS |AY AY |Aa Aa |Ae Ae |BG BG |Ar Ar |AH AH |Ax Ax |AM AM0 |Az Az |Ab Ab |Ad Ad |AO AO0 |AC AC |AE AE |AB AB |AZ AZ |At At |AT AT-1 |BL BL |An An |AI AI0 |BD BD |Aj Aj |Au Au |Av Av |Ac Ac \n".to_string();
        let mut buffered_input = Cursor::new(context_input_buffer);
        let reading_result = vw_parser.next_vowpal_with_size(&mut buffered_input);
        let (buffer, input_buffer_size) = match reading_result {
            Ok(([], _)) => panic!("CCCC"), // EOF
            Ok(buffer2) => buffer2,
            Err(_e) => panic!("DDDD"),
        };

        // ignore last newline byte
        let input_buffer_size = input_buffer_size;
        feature_buffer_translator.translate(buffer, 0);

        let mut blocks: Vec<BlockCache> = Vec::default();
        sharable_regressor.setup_cache(&feature_buffer_translator.feature_buffer, &mut blocks, true);
        println!("fw_setup_cache: {}", 0.0);

        let candidate_input_buffer = "|AP AP |AQ AQ0 |AK AK |Ao Ao |AX AXnull |AJ AJ |AU AU-1 |AL AL |AR AR |Am Am |Ap Ap |BH BH |Aq Aq0 |As As0 \n".to_string();
        let mut buffered_input = Cursor::new(&candidate_input_buffer);
        let reading_result = vw_parser
            .next_vowpal_with_cache(&mut buffered_input, input_buffer_size);

        let buffer = match reading_result {
            Ok([]) => panic!("EEEE"), // EOF
            Ok(buffer2) => buffer2,
            Err(_e) => panic!("FFFF"),
        };
        feature_buffer_translator.translate(buffer, 0);
        let prediction = sharable_regressor.predict_with_cache(&feature_buffer_translator.feature_buffer, &mut pb, blocks.as_slice());

        println!("fw_predict_with_cache: {}", prediction);
    }
}
