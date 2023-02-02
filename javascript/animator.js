const AA_GALLERY = 'animator_extension_gallery';                // Gallery object
const AA_GALLERY_CHILD = 'animator_extension_gallery_kid';      // Gallery child, as below.
const AA_INFO_LABEL = 'html_info_animator_extension';           // Info HTML object, below the gallery
const AA_ERROR_LABEL = 'html_info_x_animator_extension';        // Error label, below gallery
const AA_PROGRESS_PLACEHOLDER = 'animator_extension_gallery';   // blank HTML object above the gallery
const AA_PROC_BUTTON = 'animator_extension_procbutton';
const AA_STOP_BUTTON = 'animator_extension_stopbutton';

function start_animator() {
    rememberGallerySelection(AA_GALLERY);
    //gradioApp().getElementById(AA_ERROR_LABEL).innerHTML = ''
    var spGalleryElt = gradioApp().getElementById(AA_GALLERY);

    gradioApp().getElementById(AA_PROC_BUTTON).style.display = "none";
    //gradioApp().getElementById(AA_STOP_BUTTON).style.display = "block";

    // set id of first child of spGalleryElt to 'sp_gallery_kid',
    // required by AUTOMATIC1111 UI Logic
    spGalleryElt.children[0].id = AA_GALLERY_CHILD;
    var id = randomId();
    // requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress)
    requestProgress(id,
        gradioApp().getElementById(AA_PROGRESS_PLACEHOLDER),
        gradioApp().getElementById(AA_GALLERY_CHILD),
        function () {
            gradioApp().getElementById(AA_PROC_BUTTON).style.display = "block";
            //gradioApp().getElementById(AA_STOP_BUTTON).style.display = "none";
        },
        function (progress) {
            gradioApp().getElementById(AA_INFO_LABEL).innerHTML = progress.textinfo;
       });

    const argsToArray = args_to_array(arguments);
    argsToArray.push(argsToArray[0]);
    argsToArray[0] = id;
    return argsToArray;
}

function reenable_animator() {
    gradioApp().getElementById(AA_PROC_BUTTON).style.display = "block";
    //gradioApp().getElementById(AA_STOP_BUTTON).style.display = "none";
}

onUiUpdate(function(){
    check_gallery(AA_GALLERY)
})
