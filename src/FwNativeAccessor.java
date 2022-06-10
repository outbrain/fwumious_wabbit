import jdk.incubator.foreign.MemorySegment;
import jdk.incubator.foreign.ResourceScope;
import jdk.incubator.foreign.SegmentAllocator;
import jdk.incubator.foreign.ValueLayout;

import static com.outbrain.recs.libfw.lib_h.new_fw_predictor;
import static com.outbrain.recs.libfw.lib_h.fw_predict;

public class FwNativeAccessor implements AutoCloseable {

  private final ResourceScope scope;
  private final MemorySegment predictor;

  public FwNativeAccessor(final String command) {
    this.scope = ResourceScope.newSharedScope();
    MemorySegment commandMemorySegment = SegmentAllocator.implicitAllocator().allocateUtf8String(command);
    this.predictor = new_fw_predictor(scope, commandMemorySegment);
  }

  public float predict(String input) {
    MemorySegment cstringInput = SegmentAllocator.implicitAllocator().allocateUtf8String(input);
    MemorySegment offHeap  = SegmentAllocator.implicitAllocator().allocateArray(ValueLayout.ADDRESS, input.length());
    offHeap.set(ValueLayout.ADDRESS, 1, cstringInput);
    return fw_predict(predictor, offHeap);
  }

  @Override
  public void close() {
    this.scope.close();
  }

  public static void main(String[] args) {
    try(FwNativeAccessor fwNativeAccessor = new FwNativeAccessor("""
            fw --initial_regressor ./initial_weights.vw -t --daemon --quiet --port 26542 
            --transform "combined_widget_fromsrc_platformupper=Combine(widget_id_gen,from_src,platform_upper)()" 
            --transform "context_ctr_5log=BinnerLog(context_ctr)(0.00001,5.0)" --transform "fcap_doc_count_log=BinnerLog(fcap_doc_count)(1,5.0)" 
            --transform "from_doc_3=Weight(from_doc)(3)" --transform "interactions_count_click_sqrt=BinnerSqrtPlain(interactions_count_click)(30,1.0)" 
            --transform "interactions_count_page_view_sqrt=BinnerSqrtPlain(interactions_count_page_view)(30,1.0)" 
            --transform "logctr1a=BinnerLogRatio(interactions_count_click,interactions_count_page_view)(15,15)" 
            --transform "publisher_id_3=Weight(publisher_id)(3)" --linear ad_title_hash,from_src --linear advertiser_id,from_src 
            --linear advertiser_id,multi_adv_clicked --linear browser_family,from_src,window_w --linear browser_version,publisher_id_3,widget_id_gen 
            --linear browser_version,window_h --linear card_idx,widget_id_gen --linear from_cat,to_cat --linear from_language_lower,to_language_lower 
            --linear from_src,page_referrer --linear from_src,user_interest,widget_id_gen --linear idx,platform_upper,publisher_id_3,widget_id_gen 
            --linear idx,ui_abtest_variant_id --linear image_uuid --linear page_referrer_host --linear page_referrer_host,widget_id_gen 
            --linear requested_lst_num --linear screen_h,window_h --linear to_doc,user_interest --linear to_src,widget_id_gen --ffm_bit_precision 28 
            --ffm_field_verbose ad_title_hash,advertiser_id,campaign_id,count_clicks_6h,last_pv_hrs --ffm_field_verbose browser_version,os_version,window_w 
            --ffm_field_verbose card_idx,page_referrer_host,publisher_id_3,ui_abtest_id,widget_id_gen --ffm_field_verbose combined_widget_fromsrc_platformupper 
            --ffm_field_verbose context_ctr_5log --ffm_field_verbose conv_clicks,count_pv_24h,interactions_count_click_sqrt,last_click_hrs,multi_cat_clicked 
            --ffm_field_verbose country_v2,from_language_lower --ffm_field_verbose dma_code_v2,postal_code --ffm_field_verbose fcap_doc_count_log 
            --ffm_field_verbose from_cat --ffm_field_verbose from_doc_3 --ffm_field_verbose from_src --ffm_field_verbose idx,viewport_distance 
            --ffm_field_verbose image_uuid,to_doc --ffm_field_verbose interactions_count_page_view_sqrt --ffm_field_verbose logctr1a 
            --ffm_field_verbose multi_cmp_clicked,multi_weighted_adv_converged,user_interest --ffm_field_verbose multi_source_pv 
            --ffm_field_verbose os_family,screen_h,screen_w --ffm_field_verbose page_referrer --ffm_field_verbose region_v2 
            --ffm_field_verbose requested_lst_num,widget_id_gen --ffm_field_verbose to_doc_title_hash,to_doc_top_keyword,top_image_labels 
            --ffm_field_verbose to_src --ffm_field_verbose ui_abtest_variant_id --ffm_field_verbose user_local_hour --ffm_field_verbose window_h --ffm_k 8 
            --num_children 32
            """)) {
      String input = "|AN AN101.0.4951.61 |AH AH4456575661 |Ac Ac |Ab Ab |AO AO10 |AW AW4 |AU AU |AI AI1827 |AV AV399 |AE AEMOBILE |AP AP |AJ AJ00 |AL ALIn_App |AM AM1912 |AK AKES>MD |AT AT6 |AY AY |AX AX |AS AS3676 |AZ AZ |AG AG4843 |AR ARBROWSER |Aa Aa |AQ AQ1040032306 |AF AF63bd7f53bce110e9cbf859b6f1aa45686317944ba0771d87949f99b4b0f4ef71 |AD AD287259932 |AB AB52822534 |AC AC168496";
      float prediction = fwNativeAccessor.predict(input);
      System.out.println(prediction);
    }
  }
}
