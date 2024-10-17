#include <QCoreApplication>
#include <QCommandLineParser>

#include "qfacefusion_api.h"

class CommandLineParser
{
private:
    CommandLineParser(void){
        static CommandLineParser::GC gc;
        parser.setApplicationDescription(
            "QFacefusion example, "
            "The following parameters can be configured to start the application:");
        parser.addHelpOption();
        parser.addVersionOption();
        foreach(QString opt,commandLineMap.keys()) {
            parser.addOption(
                commandLineMap.value(opt,QCommandLineOption("defaultValue")));
        }
    }
    static CommandLineParser* self;
    QCommandLineParser parser;
    bool processApp = false;
    QMap<QString, QCommandLineOption> commandLineMap = {
        {"source",
            QCommandLineOption(
                {"s","source"},
                "set source image path",
                "source image path",
                ""
            )
        },
        {"target",
            QCommandLineOption(
                {"t","target"},
                "set target image path",
                "target image path",
                ""
            )
        },
        {"output",
            QCommandLineOption(
                {"o","output"},
                "set output image path",
                "output image path",
                ""
            )
        }
    };

public:
    void process(const QCoreApplication &app) {
        parser.process(app);
        processApp = true;
    }
    QString getOpt(const QString &optKey) const {
        if(processApp) {
            foreach(QString opt,commandLineMap.keys()) {
                if(opt == optKey){
                    QCommandLineOption dstOpt =
                        commandLineMap.value(opt,QCommandLineOption("defaultValue"));
                    if(parser.isSet(dstOpt)) {
                        return parser.value(dstOpt);
                    } else {
                        if(dstOpt.defaultValues().size() > 0)
                            return dstOpt.defaultValues().at(0);
                        else 
                            return "";
                    }
                }
            }
        }
        return "";
    }
    bool isSetOpt(const QString &optKey) const {
        if(processApp) {
            foreach(QString opt,commandLineMap.keys()) {
                if(opt == optKey){
                    QCommandLineOption dstOpt =
                        commandLineMap.value(opt,QCommandLineOption("defaultValue"));
                    return parser.isSet(dstOpt);
                }
            }
        }
        return false;
    }
    static CommandLineParser *instance() {
        if(!self) {
            self = new CommandLineParser();
        }
        return self;
    }

private:
    class GC
    {
    public:
        ~GC() {
            if (self != nullptr) {
                delete self;
                self = nullptr;
            }
        }
    };
};

CommandLineParser* CommandLineParser::self = nullptr;
#define  AppComLineParser   CommandLineParser::instance()

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);

    QCoreApplication::setApplicationName("QFacefusion example");
    QCoreApplication::setApplicationVersion("V1.0.0");

    AppComLineParser->process(application);

    QString source;
    if(AppComLineParser->isSetOpt("source")){
        source = AppComLineParser->getOpt("source");
    }
    QString target;
    if(AppComLineParser->isSetOpt("target")){
        target = AppComLineParser->getOpt("target");
    }
    QString output;
    if(AppComLineParser->isSetOpt("output")){
        output = AppComLineParser->getOpt("output");
    }

    if(source.isEmpty() || target.isEmpty() || output.isEmpty()) {
        qDebug() << "Please provide source, target and output image paths";
        return -1;
    } else {
        qDebug() << "Source image path: " << source;
        qDebug() << "Target image path: " << target;
        qDebug() << "Output image path: " << output;
        qDebug() << "Running face swap...";
    }

    QImage source_img(source);
    QImage target_img(target);
    QImage output_img;

    QFaceFusion faceSwap(MODEL_PATH);
    faceSwap.setSource(source_img);
    int ret = faceSwap.runSwap(target_img,output_img);
    if(ret < 0) {
        qDebug() << "Failed to swap faces";
    } else {
        output_img.save(output);
        qDebug() << "Face swap completed successfully";
    }

    return 0;
}
