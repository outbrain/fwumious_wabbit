use crate::version;
use clap::{App, AppSettings, Arg};

pub fn parse<'a>() -> clap::ArgMatches<'a> {
    let matches = create_expected_args().get_matches();
    matches
}

pub fn create_expected_args<'a>() -> App<'a, 'a> {
    App::new("fwumious wabbit")
        .version(version::LATEST)
        .author("Andraz Tori <atori@outbrain.com>")
        .about("Superfast Logistic Regression & Field Aware Factorization Machines")
        .setting(AppSettings::DeriveDisplayOrder)
        .arg(Arg::with_name("data")
             .long("data")
             .short("d")
             .value_name("filename")
             .help("File with input examples")
             .takes_value(true))
        .arg(Arg::with_name("quiet")
             .long("quiet")
             .help("Quiet mode, does nothing currently (as we don't output diagnostic data anyway)")
             .takes_value(false))
        .arg(Arg::with_name("predictions")
             .short("p")
             .value_name("output predictions file")
             .help("Output predictions file")
             .takes_value(true))
        .arg(Arg::with_name("cache")
             .short("c")
             .long("cache")
             .help("Use cache file")
             .takes_value(false))
        .arg(Arg::with_name("save_resume")
             .long("save_resume")
             .help("save extra state so learning can be resumed later with new data")
	     .takes_value(false))
        .arg(Arg::with_name("interactions")
             .long("interactions")
             .value_name("namespace_char,namespace_char[:value]")
             .help("Adds interactions")
             .multiple(true)
             .takes_value(true))
        .arg(Arg::with_name("linear")
             .long("linear")
             .value_name("verbose_namespace,verbose_namespace[:value]")
             .help("Adds linear feature term with optional value")
             .multiple(true)
             .takes_value(true))
        .arg(Arg::with_name("keep")
             .long("keep")
             .value_name("namespace")
             .help("Adds single features")
             .multiple(true)
             .takes_value(true))
        .arg(Arg::with_name("build_cache_without_training")
             .long("build_cache_without_training")
             .value_name("arg")
             .help("Build cache file without training the first model instance")
             .takes_value(false))

        .arg(Arg::with_name("learning_rate")
             .short("l")
             .long("learning_rate")
             .value_name("0.5")
             .help("Learning rate")
             .takes_value(true))
        .arg(Arg::with_name("ffm_learning_rate")
             .long("ffm_learning_rate")
             .value_name("0.5")
             .help("Learning rate")
             .takes_value(true))
        .arg(Arg::with_name("nn_learning_rate")
             .long("nn_learning_rate")
             .value_name("0.5")
             .help("Learning rate")
             .takes_value(true))

        .arg(Arg::with_name("minimum_learning_rate")
             .long("minimum_learning_rate")
             .value_name("0.0")
             .help("Minimum learning rate (in adaptive algos)")
             .takes_value(true))
        .arg(Arg::with_name("power_t")
             .long("power_t")
             .value_name("0.5")
             .help("How to apply Adagrad (0.5 = sqrt)")
             .takes_value(true))
        .arg(Arg::with_name("ffm_power_t")
             .long("ffm_power_t")
             .value_name("0.5")
             .help("How to apply Adagrad (0.5 = sqrt)")
             .takes_value(true))
        .arg(Arg::with_name("nn_power_t")
             .long("nn_power_t")
             .value_name("0.5")
             .help("How to apply Adagrad (0.5 = sqrt)")
             .takes_value(true))
        .arg(Arg::with_name("l2")
             .long("l2")
             .value_name("0.0")
             .help("Regularization is not supported (only 0.0 will work)")
             .takes_value(true))

        .arg(Arg::with_name("sgd")
             .long("sgd")
             .value_name("")
             .help("Disable the Adagrad, normalization and invariant updates")
             .takes_value(false))
        .arg(Arg::with_name("adaptive")
             .long("adaptive")
             .value_name("")
             .help("Use Adagrad")
             .takes_value(false))
        .arg(Arg::with_name("noconstant")
             .long("noconstant")
             .value_name("")
             .help("No intercept")
             .takes_value(false))
        .arg(Arg::with_name("link")
             .long("link")
             .value_name("logistic")
             .help("What link function to use")
             .takes_value(true))
        .arg(Arg::with_name("loss_function")
             .long("loss_function")
             .value_name("logistic")
             .help("What loss function to use")
             .takes_value(true))
        .arg(Arg::with_name("bit_precision")
             .short("b")
             .long("bit_precision")
             .value_name("18")
             .help("Size of the hash space for feature weights")
             .takes_value(true))
        .arg(Arg::with_name("hash")
             .long("hash")
             .value_name("all")
             .help("We do not support treating strings as already hashed numbers, so you have to use --hash all")
             .takes_value(true))

    // Regressor
        .arg(Arg::with_name("final_regressor")
             .short("f")
             .long("final_regressor")
             .value_name("arg")
             .help("Final regressor to save (arg is filename)")
             .takes_value(true))
        .arg(Arg::with_name("initial_regressor")
             .short("i")
             .long("initial_regressor")
             .value_name("arg")
             .help("Initial regressor(s) to load into memory (arg is filename)")
             .takes_value(true))
        .arg(Arg::with_name("testonly")
             .short("t")
             .long("testonly")
             .help("Ignore label information and just test")
             .takes_value(false))
        .arg(Arg::with_name("vwcompat")
             .long("vwcompat")
             .help("vowpal compatibility mode. Uses slow adagrad, emits warnings for non-compatible features")
             .multiple(false)
             .takes_value(false))
        .arg(Arg::with_name("convert_inference_regressor")
             .long("convert_inference_regressor")
             .value_name("arg")
             .conflicts_with("adaptive")
             .help("Inference regressor to save (arg is filename)")
             .takes_value(true))

        .arg(Arg::with_name("transform")
             .long("transform")
             .value_name("target_namespace=func(source_namespaces)(parameters)")
             .help("Create new namespace by transforming one or more other namespaces")
             .multiple(true)
             .takes_value(true))

        .arg(Arg::with_name("ffm_field")
             .long("ffm_field")
             .value_name("namespace,namespace,...")
             .help("Define a FFM field by listing namespace letters")
             .multiple(true)
             .takes_value(true))
        .arg(Arg::with_name("ffm_field_verbose")
             .long("ffm_field_verbose")
             .value_name("namespace_verbose,namespace_verbose,...")
             .help("Define a FFM field by listing verbose namespace names")
             .multiple(true)
             .takes_value(true))
        .arg(Arg::with_name("ffm_k")
             .long("ffm_k")
             .value_name("k")
             .help("Lenght of a vector to use for FFM")
             .takes_value(true))
        .arg(Arg::with_name("ffm_bit_precision")
             .long("ffm_bit_precision")
             .value_name("N")
             .help("Bits to use for ffm hash space")
             .takes_value(true))
        .arg(Arg::with_name("ffm_k_threshold")
             .long("ffm_k_threshold")
             .help("A minum gradient on left and right side to increase k")
             .multiple(false)
             .takes_value(true))
        .arg(Arg::with_name("ffm_init_center")
             .long("ffm_init_center")
             .help("Center of the initial weights distribution")
             .multiple(false)
             .takes_value(true))
        .arg(Arg::with_name("ffm_init_width")
             .long("ffm_init_width")
             .help("Total width of the initial weights distribution")
             .multiple(false)
             .takes_value(true))
        .arg(Arg::with_name("ffm_init_zero_band")
             .long("ffm_init_zero_band")
             .help("Percentage of ffm_init_width where init is zero")
             .multiple(false)
             .takes_value(true))

        .arg(Arg::with_name("nn_init_acc_gradient")
             .long("nn_init_acc_gradient")
             .help("Adagrad initial accumulated gradient for nn")
             .multiple(false)
             .takes_value(true))
        .arg(Arg::with_name("ffm_init_acc_gradient")
             .long("ffm_init_acc_gradient")
             .help("Adagrad initial accumulated gradient for ffm")
             .multiple(false)
             .takes_value(true))
        .arg(Arg::with_name("init_acc_gradient")
             .long("init_acc_gradient")
             .help("Adagrad initial accumulated gradient for ")
             .multiple(false)
             .takes_value(true))


        .arg(Arg::with_name("nn_layers")
             .long("nn_layers")
             .help("Enable deep neural network on top of LR+FFM")
             .multiple(false)
             .takes_value(true))


        .arg(Arg::with_name("nn")
             .long("nn")
             .help("Parameters of layers, for example 1:activation:relu or 2:width:20")
             .multiple(true)
             .takes_value(true))

        .arg(Arg::with_name("nn_topology")
             .long("nn_topology")
             .help("How should connections be organized - possiblities 'one' and 'two'")
             .multiple(false)
             .takes_value(true))


    // Daemon parameterts
        .arg(Arg::with_name("daemon")
             .long("daemon")
             .help("read data from port 26542")
	     .takes_value(false))
	.arg(Arg::with_name("ffm_initialization_type")
             .long("ffm_initialization_type")
             .help("Which weight initialization to consider")
             .multiple(false)
             .takes_value(true))
        .arg(Arg::with_name("port")
             .long("port")
             .value_name("arg")
             .help("port to listen on")
             .takes_value(true))
        .arg(Arg::with_name("num_children")
             .long("num_children")
             .value_name("arg (=10")
             .help("number of children for persistent daemon mode")
             .takes_value(true))
        .arg(Arg::with_name("foreground")
             .long("foreground")
             .help("in daemon mode, do not fork and run and run fw process in the foreground")
             .takes_value(false))
        .arg(Arg::with_name("prediction_model_delay")
             .conflicts_with("test_only")
             .long("prediction_model_delay")
             .value_name("examples (0)")
             .help("Output predictions with a model that is delayed by a number of examples")
             .takes_value(true))
        .arg(Arg::with_name("predictions_after")
             .long("predictions_after")
             .value_name("examples (=0)")
             .help("After how many examples start printing predictions")
             .takes_value(true))
        .arg(Arg::with_name("holdout_after")
             .conflicts_with("testonly")
             .required(false)
             .long("holdout_after")
             .value_name("examples")
             .help("After how many examples stop updating weights")
             .takes_value(true))
}
