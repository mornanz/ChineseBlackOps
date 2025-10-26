$('document').ready(function () {
    const camera = '10.98.32.1';

    $.ajax({
        url: camera+'/info',
        success: function(result){
            console.log(result);
        }
    })
});